"""Add cross-view attention fusion as an option to correct_unsw_ablation.py.

Current fusion concatenates the three view representations and passes them
through one linear layer. The views never interact: each is encoded in
isolation and the classifier sees a flat concatenation. That is the weakest
possible reading of "fusion", and it is what the earlier review questioned when
it asked why the combination is more than two techniques stacked together.

This patch adds a fusion stage where the three representations attend to one
another. Each view becomes a token in a length-3 sequence, multi-head attention
runs across those tokens, and the attended representations are then pooled and
projected. A residual connection keeps the original per-view signal, so the
attention can only add information rather than replace it.

The change is opt-in through FUSION_MODE, default "linear", so every existing
result stays reproducible. Nothing already recorded is affected.

Apply on the server:
    /usr/bin/python3 patch_cross_attention.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET = Path("/opt/ids_revision/deploy/correct_unsw_ablation.py")

ANCHOR_CONST = "AUGMENT_FRAC = 0.5"
NEW_CONST = """AUGMENT_FRAC = 0.5
# Fusion across views: "linear" concatenates and projects, "attention" lets the
# three view representations attend to one another before pooling.
FUSION_MODE = os.environ.get("FUSION_MODE", "linear")
FUSION_HEADS = 4"""

ANCHOR_CLASS = "class MultiViewEncoder(nn.Module):"

NEW_CLASS = '''class CrossViewFusion(nn.Module):
    """Let the view representations attend to one another before pooling.

    Each view arrives as one token of a length-V sequence, so multi-head
    attention runs across views rather than across features. The residual
    connection preserves the per-view signal: attention can add cross-view
    context but cannot erase what a view already encoded.
    """

    def __init__(self, d_model: int, n_views: int, nhead: int = FUSION_HEADS) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=DROPOUT, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d_model * 2, d_model),
        )
        self.project = nn.Linear(d_model * n_views, d_model)

    def forward(self, reps: list[torch.Tensor]) -> torch.Tensor:
        # (B, V, d_model): one token per view
        seq = torch.stack(reps, dim=1)
        attended, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.norm1(seq + attended)
        seq = self.norm2(seq + self.ff(seq))
        return self.project(seq.flatten(start_dim=1))


class MultiViewEncoder(nn.Module):'''

ANCHOR_INIT = "        self.fusion = nn.Linear(D_MODEL * len(view_splits), D_MODEL)"
NEW_INIT = """        if FUSION_MODE == "attention":
            self.fusion = CrossViewFusion(D_MODEL, len(view_splits))
        else:
            self.fusion = nn.Linear(D_MODEL * len(view_splits), D_MODEL)"""

ANCHOR_FWD = """        reps = [view(x[:, idx]) for view, idx in zip(self.views, self.splits)]
        return self.fusion(torch.cat(reps, dim=-1))"""
NEW_FWD = """        reps = [view(x[:, idx]) for view, idx in zip(self.views, self.splits)]
        if FUSION_MODE == "attention":
            return self.fusion(reps)
        return self.fusion(torch.cat(reps, dim=-1))"""

ANCHOR_PROTO = '"protocol": "official-split + conditional-DDPM balancing + paired-init + semantic-views"'
NEW_PROTO = ('"protocol": "official-split + conditional-DDPM balancing + paired-init + '
             'semantic-views + " + FUSION_MODE + "-fusion",\n'
             '        "fusion_mode": FUSION_MODE')

PAIRS = [
    ("fusion constants", ANCHOR_CONST, NEW_CONST),
    ("CrossViewFusion class", ANCHOR_CLASS, NEW_CLASS),
    ("encoder init", ANCHOR_INIT, NEW_INIT),
    ("encoder forward", ANCHOR_FWD, NEW_FWD),
    ("protocol string", ANCHOR_PROTO, NEW_PROTO),
]


def patch(text: str) -> str:
    if "CrossViewFusion" in text:
        raise SystemExit("already patched - aborting")
    for label, anchor, _ in PAIRS:
        if anchor not in text:
            raise SystemExit(f"anchor not found [{label}]:\n  {anchor[:90]}")
    for label, anchor, replacement in PAIRS:
        text = text.replace(anchor, replacement, 1)
        print(f"  patched: {label}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    original = TARGET.read_text(encoding="utf-8")
    patched = patch(original)
    print(f"\nlines {len(original.splitlines())} -> {len(patched.splitlines())}")

    if not args.apply:
        print("\ndry run - pass --apply to write")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(f"correct_unsw_ablation.py.bak_xattn_{stamp}")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(patched, encoding="utf-8")
    print(f"\nbackup : {backup.name}")
    print(f"written: {TARGET}")

    check = TARGET.read_text(encoding="utf-8")
    for token in ("CrossViewFusion", "FUSION_MODE", "fusion_mode", "MultiheadAttention"):
        print(f"  verify {token}: {'OK' if token in check else 'MISSING'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
