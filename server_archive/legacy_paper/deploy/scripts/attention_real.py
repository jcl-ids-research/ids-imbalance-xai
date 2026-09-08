"""Attention analysis over the real view partition.

The previous script (attention_high_standard.py) hardcodes three view lists of
15, 14 and 13 features that do not match what this revision trains. The actual
semantic partition is 14 / 16 / 12, and proto, service and state sit in view 2,
not view 1. Publishing figures from the old lists would contradict the feature
to view mapping submitted for E11, so the partition here is read from the model
checkpoint and the names from the dataset loader - the two cannot drift apart.

Tokens are individual features, so attention within a view is feature to
feature. What a reader wants from this is which features a view concentrates
on, and whether the three views specialise or duplicate each other.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from correct_unsw_ablation import (  # noqa: E402
    Classifier,
    MultiViewEncoder,
    SingleViewEncoder,
    load_unsw,
)

LABEL = {"unsw": "UNSW-NB15", "nslkdd": "NSL-KDD",
         "cicids2017": "CIC-IDS-2017", "cicddos2019": "CIC-DDoS2019"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": False})


def save(fig, outdir: Path, stem: str) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, dpi in (("pdf", None), ("png", 320)):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, format=ext, dpi=dpi, bbox_inches="tight")
        out.append(str(p))
    plt.close(fig)
    return out


def load_data(dataset: str, seed: int):
    if dataset == "unsw":
        return load_unsw("/opt/UNSW-NB15", smoke=False)
    from correct_cross_dataset import DATASETS
    return DATASETS[dataset](seed, False)


def build_model(ckpt: dict, device) -> Classifier:
    dim = int(ckpt["input_dim"])
    enc = (MultiViewEncoder(dim, ckpt["view_splits"]) if ckpt["multiview"]
           else SingleViewEncoder(dim))
    model = Classifier(enc, num_classes=int(ckpt.get("num_classes", 2)))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


class AttentionTap:
    """Read the attention maps a TransformerEncoderLayer never returns.

    Two things get in the way of the obvious approaches. The layer calls
    self_attn with need_weights=False, so a hook on the attention module sees
    None. And in eval mode PyTorch dispatches the whole layer to a fused kernel
    that never enters self_attn.forward at all, so wrapping that method
    silently captures nothing.

    A forward pre-hook does still fire, and with norm_first=False the layer
    computes norm1(x + self_attn(x, x, x)) - attention runs on the layer's raw
    input. Capturing that input and re-running self_attn on it reproduces the
    same maps exactly, without depending on any layer internals.
    """

    def __init__(self) -> None:
        self.store: dict[str, list[np.ndarray]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model: nn.Module) -> None:
        enc = model.encoder
        blocks = ([(f"view{i + 1}", v) for i, v in enumerate(enc.views)]
                  if isinstance(enc, MultiViewEncoder)
                  else [("single", enc.transformer)])
        for view_name, ft in blocks:
            for li, layer in enumerate(ft.blocks):
                key = f"{view_name}/layer{li + 1}"

                def pre_hook(module, args, _key=key):
                    x = args[0]
                    if not torch.is_tensor(x):
                        return None
                    with torch.no_grad():
                        _, w = module.self_attn(
                            x, x, x, need_weights=True,
                            average_attn_weights=True)
                    if w is not None:
                        self.store.setdefault(_key, []).append(
                            w.detach().float().mean(0).cpu().numpy())
                    return None

                self._handles.append(layer.register_forward_pre_hook(pre_hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def mean_maps(self) -> dict[str, np.ndarray]:
        return {k: np.mean(np.stack(v), axis=0) for k, v in self.store.items()}


def concentration(attn: np.ndarray) -> float:
    """Normalised entropy of the received-attention distribution.

    1.0 means every feature draws the same attention; near 0 means a view has
    collapsed onto a handful of features.
    """
    recv = attn.mean(axis=0)
    recv = recv / max(recv.sum(), 1e-12)
    nz = recv[recv > 0]
    if len(nz) <= 1:
        return 0.0
    return float(-(nz * np.log(nz)).sum() / np.log(len(recv)))


def plot_view_maps(maps, view_names, outdir, dataset, seed, layer):
    keys = [k for k in maps if k.endswith(f"/layer{layer}")]
    if not keys:
        return []
    keys.sort()
    ncol = 2 if len(keys) > 2 else len(keys)
    nrow = -(-len(keys) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.90, 7.49),
                             squeeze=False)
    for ax in axes.ravel()[len(keys):]:
        ax.axis('off')
    for ax, k in zip(axes.ravel(), keys):
        view = k.split("/")[0]
        names = view_names.get(view, [])
        m = maps[k]
        im = ax.imshow(m, cmap="viridis", aspect="auto")
        step = 1 if len(names) <= 20 else 2
        ax.set_xticks(range(0, len(names), step))
        ax.set_yticks(range(0, len(names), step))
        ax.set_xticklabels(names[::step], rotation=90, fontsize=8)
        ax.set_yticklabels(names[::step], fontsize=8)
        ax.set_title(f"{view}  ({len(names)} features)\n"
                     f"entropy {concentration(m):.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("attended to")
        ax.set_ylabel("attending from")
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.suptitle(f"{LABEL[dataset]} - self-attention within each view, "
                 f"layer {layer}, seed {seed}\n"
                 f"averaged over the test set; partition read from the checkpoint",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, f"attn_maps_{dataset}_seed{seed}_layer{layer}")


def plot_feature_importance(maps, view_names, outdir, dataset, seed, layer):
    keys = sorted(k for k in maps if k.endswith(f"/layer{layer}"))
    if not keys:
        return []
    fig, axes = plt.subplots(len(keys), 1,
                             figsize=(6.9, 2.0 * len(keys)), squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        view = k.split("/")[0]
        names = view_names.get(view, [])
        recv = maps[k].mean(axis=0)
        order = np.argsort(recv)[::-1]
        ax.bar(range(len(recv)), recv[order], color="#2E86AB",
               edgecolor="white")
        ax.set_xticks(range(len(recv)))
        ax.set_xticklabels([names[i] for i in order], rotation=75,
                           ha="right", fontsize=8)
        ax.set_ylabel("mean attention received")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{view} - {len(names)} features, "
                     f"entropy {concentration(maps[k]):.3f}",
                     fontsize=10, fontweight="bold")
    fig.suptitle(f"{LABEL[dataset]} - which features each view attends to, "
                 f"layer {layer}, seed {seed}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, f"attn_importance_{dataset}_seed{seed}_layer{layer}")


def plot_layer_evolution(maps, outdir, dataset, seed):
    views = sorted({k.split("/")[0] for k in maps})
    layers = sorted({int(k.split("layer")[1]) for k in maps})
    if len(layers) < 2:
        return []
    fig, ax = plt.subplots(figsize=(6.90, 4.29))
    for v in views:
        ys = [concentration(maps[f"{v}/layer{l}"])
              for l in layers if f"{v}/layer{l}" in maps]
        ax.plot(layers[:len(ys)], ys, marker="o", lw=1.8, label=v)
    ax.set_xticks(layers)
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("normalised attention entropy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{LABEL[dataset]} - attention concentration by depth, "
                 f"seed {seed}\n1.0 is uniform across features; "
                 f"lower means the view has specialised",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, f"attn_evolution_{dataset}_seed{seed}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unsw",
                    choices=["unsw", "nslkdd", "cicids2017", "cicddos2019"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", default="full_model")
    ap.add_argument("--models", default="results/phase1_corrected")
    ap.add_argument("--out", default="results/attention")
    ap.add_argument("--figures", default="results/figures/attention")
    ap.add_argument("--samples", type=int, default=4096)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.models, args.dataset, f"seed{args.seed}",
                             f"model_{args.variant}.pt")
    if not os.path.exists(ckpt_path):
        print(f"checkpoint missing: {ckpt_path}")
        return 1
    ckpt = torch.load(ckpt_path, map_location=device)
    if not ckpt.get("multiview"):
        print(f"{args.variant} is single-view; nothing to compare across views")
        return 1

    model = build_model(ckpt, device)
    data = load_data(args.dataset, args.seed)
    names = list(data.feature_names)
    view_names = {f"view{i + 1}": [names[j] for j in idx]
                  for i, idx in enumerate(ckpt["view_splits"])}
    print(f"[{args.dataset} seed{args.seed} {args.variant}] view sizes: "
          f"{[len(v) for v in view_names.values()]}")
    for v, fs in view_names.items():
        print(f"  {v}: {fs}")

    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(data.X_test), min(args.samples, len(data.X_test)),
                     replace=False)
    X = torch.FloatTensor(data.X_test[idx])

    tap = AttentionTap()
    tap.attach(model)
    with torch.no_grad():
        for i in range(0, len(X), 512):
            model(X[i:i + 512].to(device))
    tap.detach()

    maps = tap.mean_maps()
    if not maps:
        print("no attention captured")
        return 1
    print(f"captured {len(maps)} attention maps")

    n_layers = max(int(k.split("layer")[1]) for k in maps)
    figdir = Path(args.figures)
    written = []
    written += plot_view_maps(maps, view_names, figdir, args.dataset,
                              args.seed, n_layers)
    written += plot_feature_importance(maps, view_names, figdir, args.dataset,
                                       args.seed, n_layers)
    written += plot_layer_evolution(maps, figdir, args.dataset, args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset, "seed": args.seed, "variant": args.variant,
        "n_samples": int(len(X)),
        "view_partition": view_names,
        "entropy": {k: concentration(v) for k, v in maps.items()},
        "received_attention": {
            k: dict(zip(view_names.get(k.split("/")[0], []),
                        v.mean(axis=0).round(6).tolist()))
            for k, v in maps.items()},
    }
    dst = outdir / f"attention_{args.dataset}_seed{args.seed}_{args.variant}.json"
    dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {dst}")
    print(f"[OK] {len(written)} figure files -> {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
