"""
Correct UNSW-NB15 ablation: official split + real conditional DDPM augmentation.

Protocol (fixes the invalid legacy runs):
  - Official split: UNSW_NB15_training-set.csv (175,341) = train; UNSW_NB15_testing-set.csv (82,332) = test.
  - All encoders, median imputation, and scaling are fit on TRAINING data only.
  - Dev/val split is taken from the training partition only (stratified).
  - A class-conditional DDPM is trained per class on the subtrain only.
  - Augmentation doubles the subtrain scale while preserving class proportions.
  - Paired initialization: full/wo_diffusion share one init; wo_multiview/baseline share another.
  - Multi-view = 3 per-view transformers (feature-as-token); single-view = 1 transformer (all features as tokens).

Run:
  /usr/bin/python3 correct_unsw_ablation.py --seed 42 --out-dir /opt/ids_revision/results/ablation_repeat_correct
  /usr/bin/python3 correct_unsw_ablation.py --smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# Semantic feature -> view partition (paper Table 2 intent), covering all 42
# UNSW-NB15 features exactly once. View sizes 14 / 16 / 12.
SEMANTIC_VIEWS: dict[str, tuple[str, ...]] = {
    "view1_flow_rate": (
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
        "sloss", "dloss", "sinpkt", "dinpkt", "smean", "dmean",
    ),
    "view2_connection_state": (
        "proto", "service", "state", "sttl", "dttl", "swin", "dwin",
        "ct_state_ttl", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_sm_ips_ports",
    ),
    "view3_service_payload": (
        "sjit", "djit", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat",
        "trans_depth", "response_body_len", "ct_flw_http_mthd", "is_ftp_login",
        "ct_ftp_cmd",
    ),
}

DEFAULT_DATA_DIR = "/opt/UNSW-NB15"
DEFAULT_OUT_DIR = "/opt/ids_revision/results/ablation_repeat_correct"
D_MODEL = 128
NHEAD = 8
N_LAYERS = 4
DIM_FF = 512
DROPOUT = 0.1
N_VIEWS = 3
DDPM_TIMESTEPS = 500
DDPM_HIDDEN = (256, 256, 256)
CLASSIFIER_EPOCHS = 30
CLASSIFIER_BATCH = 256
CLASSIFIER_PATIENCE = 5
DDPM_EPOCHS = 1000
DDPM_BATCH = 1024
AUGMENT_FRAC = 0.5
# Expansion cap: prevent extreme synthetic sample generation (e.g., 100x for tiny classes).
# Set to 15 after cost analysis (CIC-DDoS2019 needs 16.12x, 15x achieves 1.07 imbalance ratio).
CAP_EXPANSION = 15


@dataclass(frozen=True, slots=True)
class VariantMetrics:
    accuracy: float
    f1: float
    f1_macro: float
    precision: float
    recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "acc": self.accuracy,
            "f1": self.f1,
            "f1_macro": self.f1_macro,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass(frozen=True, slots=True)
class LoadedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    feature_names: tuple[str, ...]


def semantic_view_splits(feature_names: tuple[str, ...]) -> list[list[int]]:
    """Map SEMANTIC_VIEWS names to column indices, failing loudly on any mismatch."""
    index_of = {name: i for i, name in enumerate(feature_names)}
    assigned: list[list[int]] = []
    seen: set[str] = set()
    for view, names in SEMANTIC_VIEWS.items():
        missing = [n for n in names if n not in index_of]
        if missing:
            raise KeyError(f"{view}: features absent from dataset: {missing}")
        duplicated = [n for n in names if n in seen]
        if duplicated:
            raise ValueError(f"{view}: features assigned to more than one view: {duplicated}")
        seen.update(names)
        assigned.append([index_of[n] for n in names])
    uncovered = [n for n in feature_names if n not in seen]
    if uncovered:
        raise ValueError(f"features not assigned to any view: {uncovered}")
    return assigned


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :]


class _FeatureTransformer(nn.Module):
    """Shared per-view transformer block (feature-as-token)."""

    def __init__(self, seq_len: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.projector = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True, activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V) -> tokens
        b, v = x.shape
        tok = x.reshape(b * v, 1)
        tok = self.projector(tok).reshape(b, v, -1)
        tok = self.pos_enc(tok)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        return tok.mean(dim=1)  # (B, d_model)


class MultiViewEncoder(nn.Module):
    def __init__(self, input_dim: int, view_splits: list[list[int]]) -> None:
        super().__init__()
        self.splits = view_splits
        self.view_dims = [len(s) for s in view_splits]
        self.views = nn.ModuleList(
            [
                _FeatureTransformer(vd, D_MODEL, NHEAD, N_LAYERS, DIM_FF, DROPOUT)
                for vd in self.view_dims
            ]
        )
        self.fusion = nn.Linear(D_MODEL * len(view_splits), D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reps = [view(x[:, idx]) for view, idx in zip(self.views, self.splits)]
        return self.fusion(torch.cat(reps, dim=-1))


class SingleViewEncoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.transformer = _FeatureTransformer(input_dim, D_MODEL, NHEAD, N_LAYERS, DIM_FF, DROPOUT)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.transformer(x))


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class MLPDDPM(nn.Module):
    """Class-specific DDPM with an MLP epsilon-prediction denoiser."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.timesteps = DDPM_TIMESTEPS
        betas = torch.linspace(1e-4, 0.02, DDPM_TIMESTEPS)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        layers: list[nn.Module] = []
        prev = input_dim + 1
        for h in DDPM_HIDDEN:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.denoiser = nn.Sequential(*layers)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        sab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        somab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        xt = sab * x0 + somab * noise
        t_embed = (t.float() / self.timesteps).unsqueeze(-1)
        pred = self.denoiser(torch.cat([xt, t_embed], dim=-1))
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(n, self.input_dim, device=device)
        for t in reversed(range(self.timesteps)):
            t_t = torch.full((n,), t, device=device, dtype=torch.long)
            t_embed = (t_t.float() / self.timesteps).unsqueeze(-1)
            pred = self.denoiser(torch.cat([x, t_embed], dim=-1))
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            x = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred
            ) + torch.sqrt(beta_t) * z
            # A few reverse trajectories diverge to +-inf, and inf - inf then
            # becomes NaN, which silently propagates to every downstream step.
            # Sanitize and bound each step: in quantile space the target is
            # N(0,1), so anything past +-5 sigma is already meaningless.
            x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
            x = torch.clamp(x, -5.0, 5.0)
        return x


def load_unsw(data_dir: str, smoke: bool = False) -> LoadedData:
    tr = pd.read_csv(os.path.join(data_dir, "UNSW_NB15_training-set.csv"))
    te = pd.read_csv(os.path.join(data_dir, "UNSW_NB15_testing-set.csv"))
    if smoke:
        tr = tr.sample(n=4000, random_state=0)
        te = te.sample(n=2000, random_state=0)

    for df in (tr, te):
        df.drop(columns=[c for c in ["id", "Unnamed: 0"] if c in df.columns],
                inplace=True, errors="ignore")

    feat_cols = [c for c in tr.columns if c not in ("label", "attack_cat")]
    cat_cols = [c for c in feat_cols if tr[c].dtype == object]

    for c in cat_cols:
        cats = sorted(tr[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(cats)}
        tr[c] = tr[c].astype(str).map(mapping).astype(np.float32)
        te[c] = te[c].astype(str).map(mapping).fillna(-1.0).astype(np.float32)

    medians = tr[feat_cols].median(numeric_only=True)
    tr = tr.fillna(medians)
    te = te.fillna(medians)

    X_tr = tr[feat_cols].astype(np.float32).to_numpy()
    X_te = te[feat_cols].astype(np.float32).to_numpy()
    y_tr = tr["label"].values.astype(np.int64)
    y_te = te["label"].values.astype(np.int64)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return LoadedData(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                      n_features=X_tr.shape[1], feature_names=tuple(feat_cols))


def train_ddpm(model: MLPDDPM, X: np.ndarray, device: torch.device,
               epochs: int, label: str = "") -> None:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    xt = torch.FloatTensor(X).to(device)
    n = xt.size(0)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total = 0.0
        cnt = 0
        for i in range(0, n, DDPM_BATCH):
            batch = xt[perm[i : i + DDPM_BATCH]]
            opt.zero_grad()
            loss = model(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            cnt += 1
        if ep % 200 == 0 or ep == 1:
            print(f"    [DDPM {label}] epoch {ep}/{epochs} loss={total / cnt:.5f}", flush=True)


def _as_tensor(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.FloatTensor(X), torch.LongTensor(y)


def _evaluate(model: Classifier, X: np.ndarray, y: np.ndarray,
              device: torch.device, batch: int) -> VariantMetrics:
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.FloatTensor(X[i : i + batch]).to(device)
            preds.extend(model(xb).argmax(dim=1).cpu().numpy().tolist())
    acc = accuracy_score(y, preds)
    p, r, f1, _ = precision_recall_fscore_support(y, preds, average="weighted", zero_division=0)
    _, _, f1_m, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
    return VariantMetrics(accuracy=float(acc), f1=float(f1), f1_macro=float(f1_m),
                          precision=float(p), recall=float(r))


def _train_classifier(model: Classifier, X: np.ndarray, y: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, device: torch.device,
                      epochs: int, batch: int, patience: int) -> Classifier:
    model = model.to(device)
    Xt, yt = _as_tensor(X, y)
    Xv, yv = _as_tensor(X_val, y_val)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_f1 = 0.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for ep in range(1, epochs + 1):
        model.train()
        # Shuffle every epoch: augmented data is concatenated class-block-wise,
        # so sequential iteration would end each epoch on a single-class run and
        # collapse the model to the majority class.
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i : i + batch]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        vp: list[int] = []
        with torch.no_grad():
            for i in range(0, len(Xv), batch):
                vp.extend(model(Xv[i : i + batch].to(device)).argmax(dim=1).cpu().numpy().tolist())
        _, _, f1, _ = precision_recall_fscore_support(y_val, vp, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


_DISCRETE_CORRECTION = os.environ.get("DISCRETE_CORRECTION", "1") != "0"
try:
    from discrete_correction import correct_discrete, detect_discrete_columns
except ImportError:  # helper absent -> behave exactly as before
    _DISCRETE_CORRECTION = False
    correct_discrete = None
    detect_discrete_columns = None


def rebalance(ddpm_models: dict[int, MLPDDPM], X: np.ndarray, y: np.ndarray,
              qt: QuantileTransformer, device: torch.device,
              seed: int, cap: int = CAP_EXPANSION) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    """Balancing module as described in the paper: reduce the dominant class and
    augment minority classes with diffusion-generated samples until every class
    holds the same number of records.

    Target per class is the mean class count, so majority classes are
    undersampled and minority classes are topped up by the DDPM. An expansion
    cap prevents extreme synthetic generation (e.g., 100x for tiny multiclass
    categories), trading perfect balance for fidelity credibility.

    The total training size is preserved (or reduced if cap triggers), which
    keeps the ablation contrast about *balancing* rather than about data volume.

    Returns (X_balanced, y_balanced, report, X_synthetic, y_synthetic). The
    report records the class distribution before and after - this is the
    evidence the editor asked for. The synthetic arrays are returned separately
    so fidelity metrics can be computed without re-generating them.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    mean_target = int(round(counts.mean()))

    # Discreteness is a property of the feature, so it is decided once on the
    # full training matrix. Continuous diffusion smears these columns onto their
    # modal value; the KS test cannot see it because >99% of the mass already
    # sits there. See discrete_correction.py.
    discrete_flags = (detect_discrete_columns(X)
                      if _DISCRETE_CORRECTION else None)
    if discrete_flags is not None:
        print(f"    [DISCRETE] correcting {int(discrete_flags.sum())}/"
              f"{X.shape[1]} discrete columns", flush=True)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    syn_parts: list[np.ndarray] = []
    syn_labels: list[np.ndarray] = []
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    actions: dict[str, str] = {}
    expansion: dict[str, float] = {}

    for cls, n_real in zip(classes.tolist(), counts.tolist()):
        cls = int(cls)
        before[str(cls)] = int(n_real)
        real = X[y == cls]

        if n_real >= mean_target:
            # Majority class: undersample to mean target
            target_c = mean_target
            keep = rng.choice(n_real, size=target_c, replace=False)
            X_parts.append(real[keep])
            y_parts.append(np.full(target_c, cls, dtype=np.int64))
            actions[str(cls)] = f"undersampled {n_real} -> {target_c}"
            after[str(cls)] = target_c
            expansion[str(cls)] = round(target_c / n_real, 4)
            continue

        # Minority class: augment, but cap expansion ratio to prevent extreme generation
        target_c = min(mean_target, n_real * cap)
        n_syn = target_c - n_real
        syn_q = ddpm_models[cls].sample(n_syn, device).cpu().numpy().astype(np.float32)
        syn_q = np.clip(syn_q, -4.0, 4.0)
        syn = qt.inverse_transform(syn_q).astype(np.float32)
        syn = np.clip(syn, real.min(axis=0), real.max(axis=0)).astype(np.float32)
        if not np.isfinite(syn).all():
            bad = int((~np.isfinite(syn)).sum())
            raise FloatingPointError(
                f"class {cls}: {bad} non-finite synthetic values after clipping")

        # Rank-matched marginal correction for discrete attributes. The rank
        # ordering the diffusion model learned is preserved, so dependence with
        # the continuous columns survives, while the marginal is restored to the
        # real one - which keeps attack-defining flags (root_shell,
        # num_failed_logins, SYN Flag Count) from collapsing to a constant.
        # Applied before the concatenation so training data and samples.npz both
        # carry the corrected values.
        if discrete_flags is not None:
            syn, _ = correct_discrete(real, syn, discrete_flags)
            syn = syn.astype(np.float32)

        X_parts.append(np.concatenate([real, syn]))
        y_parts.append(np.full(target_c, cls, dtype=np.int64))
        syn_parts.append(syn)
        syn_labels.append(np.full(n_syn, cls, dtype=np.int64))
        cap_hit = " (CAP HIT)" if target_c < mean_target else ""
        actions[str(cls)] = f"augmented {n_real} + {n_syn} synthetic -> {target_c}{cap_hit}"
        after[str(cls)] = target_c
        expansion[str(cls)] = round(target_c / n_real, 4)
        print(f"    [BAL class {cls}] real n={n_real} mean={real.mean():.3f} "
              f"std={real.std():.3f} | syn n={n_syn} mean={syn.mean():.3f} "
              f"std={syn.std():.3f}", flush=True)

    X_bal = np.concatenate(X_parts)
    y_bal = np.concatenate(y_parts)
    perm = rng.permutation(len(X_bal))

    X_syn = (np.concatenate(syn_parts) if syn_parts
             else np.empty((0, X.shape[1]), dtype=np.float32))
    y_syn = (np.concatenate(syn_labels) if syn_labels
             else np.empty(0, dtype=np.int64))

    after_counts = np.array([after[str(c)] for c in classes], dtype=np.int64)
    imbalance_after = round(float(after_counts.max() / after_counts.min()), 4)
    
    report = {
        "mean_target": mean_target,
        "expansion_cap": cap,
        "before": before,
        "after": after,
        "actions": actions,
        "expansion_ratio": expansion,
        "synthetic_total": int(len(X_syn)),
        "total_before": int(len(X)),
        "total_after": int(len(X_bal)),
        "imbalance_ratio_before": round(float(counts.max() / counts.min()), 4),
        "imbalance_ratio_after": imbalance_after,
    }
    print(f"    [BAL] mean_target={mean_target}/class  cap={cap}x  "
          f"imbalance {report['imbalance_ratio_before']:.2f} -> {imbalance_after:.2f}  "
          f"before={before}  after={after}", flush=True)
    return X_bal[perm], y_bal[perm], report, X_syn, y_syn


def augment(ddpm_models: dict[int, MLPDDPM], X: np.ndarray, y: np.ndarray,
            qt: QuantileTransformer, device: torch.device,
            frac: float = AUGMENT_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """Sample in quantile (Gaussianized) space, inverse-transform back, then mix.

    X is in StandardScaler space; DDPMs are trained on quantile-transformed data
    so heavy-tailed features are Gaussianized. Synthetic samples are inverse
    quantile-transformed before mixing so they match the real scaled distribution.
    """
    X_list = [X]
    y_list = [y]
    for cls, model in ddpm_models.items():
        mask = y == cls
        n_real = int(mask.sum())
        n = max(1, int(n_real * frac))
        syn_q = model.sample(n, device).cpu().numpy().astype(np.float32)
        # Clip in quantile space before inverting: norm.ppf overflows to +-inf
        # near the 0/1 boundaries, which previously produced NaN rows that
        # silently poisoned training. +-4 sigma keeps 99.99% of N(0,1) mass.
        syn_q = np.clip(syn_q, -4.0, 4.0)
        syn = qt.inverse_transform(syn_q).astype(np.float32)
        real = X[mask]
        # Constrain to the real per-feature envelope so synthetic rows stay
        # inside the observed data range.
        syn = np.clip(syn, real.min(axis=0), real.max(axis=0)).astype(np.float32)
        if not np.isfinite(syn).all():
            bad = int((~np.isfinite(syn)).sum())
            raise FloatingPointError(
                f"class {cls}: {bad} non-finite synthetic values after clipping"
            )
        print(f"    [AUG class {cls}] real n={n_real} mean={real.mean():.3f} std={real.std():.3f} "
              f"| syn n={n} mean={syn.mean():.3f} std={syn.std():.3f}", flush=True)
        X_list.append(syn)
        y_list.append(np.full(n, cls, dtype=np.int64))
    return np.concatenate(X_list), np.concatenate(y_list)


def run_seed(seed: int, data_dir: str, out_dir: str, smoke: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_unsw(data_dir, smoke=smoke)

    X_sub, X_val, y_sub, y_val = train_test_split(
        data.X_train, data.y_train, test_size=0.1, random_state=seed, stratify=data.y_train
    )

    ddpm_epochs = 2 if smoke else DDPM_EPOCHS
    cls_epochs = 2 if smoke else CLASSIFIER_EPOCHS
    cls_batch = 128 if smoke else CLASSIFIER_BATCH

    # Gaussianize features via quantile transform (fit on subtrain only),
    # then train class-conditional DDPMs in the quantile space.
    qt = QuantileTransformer(
        output_distribution="normal", random_state=seed,
        n_quantiles=min(1000, len(X_sub)),
    )
    X_sub_q = qt.fit_transform(X_sub)

    ddpm_models: dict[int, MLPDDPM] = {}
    for cls in np.unique(y_sub):
        X_cls_q = X_sub_q[y_sub == cls]
        m = MLPDDPM(data.n_features)
        train_ddpm(m, X_cls_q, device, ddpm_epochs, label=str(int(cls)))
        ddpm_models[int(cls)] = m

    # Balancing module (paper Sec. "rebalancing strategy"): undersample the
    # dominant class and top up minority classes with DDPM samples until every
    # class is equally represented. balance_report is the evidence the editor
    # asked for when noting the distribution "does not appear balanced".
    X_bal, y_bal, balance_report, _X_syn, _y_syn = rebalance(
        ddpm_models, X_sub, y_sub, qt, device, seed
    )

    # Paired initialization. Multi-view uses the paper's semantic partition.
    view_splits = semantic_view_splits(data.feature_names)
    print(f"  [VIEWS] sizes={[len(s) for s in view_splits]}", flush=True)
    full_model = Classifier(MultiViewEncoder(data.n_features, view_splits))
    wo_diff = copy.deepcopy(full_model)
    wo_mv = Classifier(SingleViewEncoder(data.n_features))
    wo_both = copy.deepcopy(wo_mv)

    # Variants with the balancing module see X_bal; variants without it see the
    # untouched imbalanced subtrain. This is the contrast the editor asked to
    # isolate (balancing module vs multi-view architecture).
    _train_classifier(full_model, X_bal, y_bal, X_val, y_val, device, cls_epochs, cls_batch, CLASSIFIER_PATIENCE)
    _train_classifier(wo_diff, X_sub, y_sub, X_val, y_val, device, cls_epochs, cls_batch, CLASSIFIER_PATIENCE)
    _train_classifier(wo_mv, X_bal, y_bal, X_val, y_val, device, cls_epochs, cls_batch, CLASSIFIER_PATIENCE)
    _train_classifier(wo_both, X_sub, y_sub, X_val, y_val, device, cls_epochs, cls_batch, CLASSIFIER_PATIENCE)

    variants = {
        "full_model": _evaluate(full_model, data.X_test, data.y_test, device, cls_batch),
        "wo_diffusion": _evaluate(wo_diff, data.X_test, data.y_test, device, cls_batch),
        "wo_multiview": _evaluate(wo_mv, data.X_test, data.y_test, device, cls_batch),
        "baseline": _evaluate(wo_both, data.X_test, data.y_test, device, cls_batch),
    }

    result = {
        "seed": seed,
        "protocol": "official-split + conditional-DDPM balancing + paired-init + semantic-views",
        "view_partition": {k: list(v) for k, v in SEMANTIC_VIEWS.items()},
        "sizes": {
            "train": int(len(X_sub)),
            "val": int(len(X_val)),
            "test": int(len(data.X_test)),
            "balanced_train": int(len(X_bal)),
        },
        "balance_report": balance_report,
        "variants": {k: v.as_dict() for k, v in variants.items()},
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"unsw_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] saved {out_path}")
    return result


def ddpm_check(data_dir: str, seed: int) -> None:
    """Train the class-1 DDPM in quantile space and report post-inverse-transform stats."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_unsw(data_dir)
    X_sub, _, y_sub, _ = train_test_split(
        data.X_train, data.y_train, test_size=0.1, random_state=seed, stratify=data.y_train
    )
    qt = QuantileTransformer(output_distribution="normal", random_state=seed,
                             n_quantiles=min(1000, len(X_sub)))
    X_sub_q = qt.fit_transform(X_sub)
    cls = 1
    X_cls_q = X_sub_q[y_sub == cls]
    X_cls = X_sub[y_sub == cls]
    print(f"[CHECK] class {cls}: {X_cls.shape[0]} samples")
    m = MLPDDPM(data.n_features)
    train_ddpm(m, X_cls_q, device, DDPM_EPOCHS, label=f"check-cls{cls}")
    syn_q = m.sample(5000, device).cpu().numpy().astype(np.float32)
    syn = qt.inverse_transform(syn_q).astype(np.float32)
    print(f"[CHECK] real mean={X_cls.mean():.3f} std={X_cls.std():.3f} "
          f"min={X_cls.min():.3f} max={X_cls.max():.3f}")
    print(f"[CHECK] syn  mean={syn.mean():.3f} std={syn.std():.3f} "
          f"min={syn.min():.3f} max={syn.max():.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ddpm-check", action="store_true")
    args = parser.parse_args()

    if args.ddpm_check:
        ddpm_check(args.data_dir, args.seed)
        return 0

    t0 = time.time()
    result = run_seed(args.seed, args.data_dir, args.out_dir, args.smoke)
    print(json.dumps(result["variants"], indent=2))
    print(f"Done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
