#!/usr/bin/env python3
"""Fix old checkpoints: add missing buffers to CosineConv2d layers."""
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import fsod.modules  # noqa — ensures current CosineConv2d is importable

REQUIRED_BUFFERS = ["weight_prior", "weight_prior_mask", "fixed_blend_alpha"]


def fix_checkpoint(ckpt_path: Path) -> bool:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = ckpt["model"]
    fixed = 0

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name != "CosineConv2d":
            continue
        for buf_name in REQUIRED_BUFFERS:
            if not hasattr(module, buf_name):
                print(f"  FIX: {name}.{buf_name} — adding missing buffer")
                fixed += 1
            # Ensure buffer exists with correct shape
            existing = getattr(module, buf_name, None)
            if buf_name == "weight_prior":
                expected = torch.zeros(module.out_channels, module.in_channels)
            elif buf_name == "weight_prior_mask":
                expected = torch.zeros(module.out_channels, dtype=torch.bool)
            elif buf_name == "fixed_blend_alpha":
                expected = torch.tensor(0.5)
            else:
                continue
            setattr(module, buf_name, expected)

    if fixed > 0:
        torch.save(ckpt, str(ckpt_path))
    return fixed > 0


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        # Find all best.pt files under runs/
        base = Path("runs")
        paths = list(base.rglob("**/best.pt"))

    print(f"Checking {len(paths)} checkpoints...\n")
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if fix_checkpoint(p):
            print(f"  ✅ Fixed: {p}\n")
        else:
            print(f"  ⏭️  Skipped (no fix needed): {p}\n")


if __name__ == "__main__":
    main()
