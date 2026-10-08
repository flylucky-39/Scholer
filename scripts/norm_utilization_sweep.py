"""Post-hoc feature-norm utilization sweep on COCO Cosine+Proto checkpoint.

Hypothesis: YOLO's linear cls head scores = ||w||*||x||*cos(theta) — it uses the
feature norm as an implicit objectness signal. The cosine head discards ||x||
(few-shot robustness by design). On COCO (20 novel classes, cluttered scenes)
the discarded norm information poisons within-class ranking (FP decomposition:
2.3 FPs/img vs 0.03 for standard). This sweep restores norm utilization
post-hoc:  score = exp(s_c) * cos(x, w_c) * (||x||/c)^alpha + bias
with alpha in {0, 0.25, 0.5, 0.75, 1.0}. alpha=0 reproduces the trained model;
alpha=1 approximates linear-head norm usage with cosine-direction weights.

Usage:
  python scripts/norm_utilization_sweep.py
"""
from __future__ import annotations

import sys
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401
from ultralytics import YOLO

NOVEL = {'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'boat',
         'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 'couch',
         'potted plant', 'dining table', 'tv'}

CKPT = PROJECT_ROOT / 'runs' / 'coco_fsod_10shot_New' / 'novel_finetune_cosine_proto' / 'weights' / 'best.pt'
DATA = PROJECT_ROOT / 'data' / 'coco_fsod_10shot'
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
# reference norm (measured: penultimate feature norms ~15-20); keeps score scale stable
REF_NORM = 17.0


def patched_forward(self, x):
    w_norm = F.normalize(self.weight, dim=1)
    x_norm = F.normalize(x, dim=1)
    cos_sim = F.conv2d(x_norm, w_norm.unsqueeze(-1).unsqueeze(-1))
    if self._alpha > 0:
        xn = x.norm(dim=1, keepdim=True).clamp(min=1e-6)
        cos_sim = cos_sim * xn.pow(self._alpha) / (REF_NORM ** self._alpha)
    if self.has_bg_suppression:
        bg_norm = F.normalize(self.bg_proto, dim=0)
        bg_cos = F.conv2d(x_norm, bg_norm.view(1, self.in_channels, 1, 1))
        cos_sim = cos_sim - self.bg_gamma * bg_cos
    return cos_sim * self.scale.exp().view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def main():
    out = []
    print(f"{'alpha':>6}{'novel mAP50':>13}{'mAP50-95':>10}{'P':>8}{'R':>8}")
    print('-' * 50)
    for alpha in ALPHAS:
        model = YOLO(str(CKPT))
        n_patched = 0
        for name, mod in model.model.named_modules():
            if mod.__class__.__name__ == 'CosineConv2d':
                if not hasattr(mod, 'bg_proto'):
                    mod.register_buffer('bg_proto', torch.zeros(mod.in_channels))
                    mod.register_buffer('bg_gamma', torch.tensor(0.0))
                mod._alpha = alpha
                mod.forward = types.MethodType(patched_forward, mod)
                n_patched += 1
        assert n_patched == 3, f"expected 3 CosineConv2d, patched {n_patched}"
        met = model.val(data=str(DATA / 'coco_fsod_finetune.yaml'), split='test',
                        imgsz=640, device=0, verbose=False, plots=False)
        ap50 = met.box.ap50
        ap = met.box.ap
        cls_idx = met.box.ap_class_index
        names = met.names
        sel = [i for i, c in enumerate(cls_idx) if names[int(c)] in NOVEL]
        m50 = float(np.mean([ap50[i] for i in sel]))
        r = dict(alpha=alpha, novel_map50=m50,
                 map50_95=float(np.mean([ap[i] for i in sel])),
                 P=float(met.box.mp), R=float(met.box.mr))
        out.append(r)
        print(f"{alpha:>6.2f}{m50:>13.4f}{r['map50_95']:>10.4f}{r['P']:>8.3f}{r['R']:>8.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    dst = PROJECT_ROOT.parent / 'final_experiments' / 'results' / 'norm_utilization_sweep_coco.json'
    dst.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"\nsaved → {dst}")


if __name__ == '__main__':
    main()
