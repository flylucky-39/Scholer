"""
Preload script: patches ultralytics.nn.tasks.parse_model to handle FSODDetect (subclass of Detect).
Also sets up PYTHONPATH and registers necessary module references.

Usage:
    python preload_train_fsod.py --config ... [other train_fsod.py args]
"""
import sys
import os
from pathlib import Path

# Project layout:
#   /root/epfs/07_FSOD_LLM/fsod/   <- we run from here
#     third_party/ultralytics/      <- custom ultralytics
#     fsod/                         <- actual Python package
#     scripts/train_fsod.py
PROJECT_DIR = Path.cwd()  # /root/epfs/07_FSOD_LLM/fsod/
os.environ["PYTHONPATH"] = f"{PROJECT_DIR / 'third_party' / 'ultralytics'}:{PROJECT_DIR}:" + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(PROJECT_DIR / "third_party" / "ultralytics"))
sys.path.insert(0, str(PROJECT_DIR))

# ============================================================
# Step 1: Patch parse_model BEFORE importing ultralytics.YOLO
# ============================================================
# We need to replace the frozenset inside parse_model to include FSODDetect.
# The frozenset is a "constant" inside the function (created at definition time).
# We monkey-patch by wrapping parse_model after it's defined.
import ultralytics.nn.tasks as tasks_module
import fsod.modules.cosine_head as cosine_head

FSODDetect_cls = cosine_head.FSODDetect

# Register for YAML name resolution
tasks_module.FSODDetect = FSODDetect_cls

# Now patch parse_model to handle FSODDetect like Detect
_orig_parse_model = tasks_module.parse_model

def _patched_parse_model(d, ch, nc, device="cpu", verbose=False, *, depth=1.0, width=1.0, max_channels=512, scale="", end2end=False, reg_max=16, legacy=True):
    """Patched parse_model that handles FSODDetect like Detect."""
    import torch
    from ultralytics.nn.modules import (
        AIFI, C1, C2, C2PSA, C2PSA2, C2f, C2fAttn, C2fCIB, C3, C3TR, C3k2, C3x,
        Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, PWConv,
        RepNCSPELAN4, SCDown, Segment, WorldDetect, v10Detect, A2C2f, C2fPSA2,
    )

    # Re-use the base_modules and repeat_modules frozensets from original
    # We'll replicate the original logic but handle FSODDetect
    from ultralytics.nn.modules import (
        AConv, ADown, Bottleneck, BottleneckCSP, C1, C2, C2PSA, C2PSA2, C2f,
        C2fAttn, C2fCIB, C2fPSA, C3, C3TR, C3k2, C3x, CBFuse, CBLinear,
        C2fPSA2, C2PSA2,
        Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
        ELAN1, Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, ImagePoolingAttn,
        OBB, OBB26, PSA, PWConv, Pose, Pose26, RepC3, RepNCSPELAN4, RTDETRDecoder,
        SCDown, SPP, SPPELAN, SPPF, Segment, Segment26, SemanticSegment, TorchVision,
        WorldDetect, YOLOEDetect, YOLOESegment, YOLOESegment26,
    )
    from ultralytics.utils import LOGGER, make_divisible
    import numpy as np
    import torch.nn
    import torchvision
    import ast
    import contextlib

    # Try to import the scale-regularized variant too
    try:
        from fsod.modules.scale_regularization import FSODDetect_ScaleReg
    except ImportError:
        FSODDetect_ScaleReg = None

    # Register our custom modules in the local globals
    FSODDetect = FSODDetect_cls

    # Build the detect-like frozenset, including FSODDetect
    detect_like = frozenset({
        Detect, WorldDetect, YOLOEDetect, Segment, Segment26,
        YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26,
        FSODDetect,
    })
    if FSODDetect_ScaleReg is not None:
        detect_like = frozenset(detect_like | {FSODDetect_ScaleReg})

    base_modules = frozenset(
        {
            Conv, Conv2, LightConv, RepConv, DWConv, PWConv, ConvTranspose,
            Focus, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF,
            C2fPSA, C2PSA, C2PSA2, C2fPSA2, DWConv, Focus, BottleneckCSP,
            C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN,
            C2fAttn, C3, C3TR, C3Ghost, torch.nn.ConvTranspose2d,
            DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, A2C2f,
        }
    )
    repeat_modules = frozenset(
        {
            BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost,
            C3x, RepC3, C2fPSA, C2fCIB, C2PSA, C2PSA2, C2fPSA2, A2C2f,
        }
    )

    # Register more needed names
    LightConv = Conv2  # alias

    # Replicate the full parse_model logic
    scales = {"n": 0.50, "s": 0.50, "m": 1.00, "l": 1.00, "x": 1.50}
    depth, width, max_channels = (
        scales[scale] if scale in scales else 1,
        scales[scale] if scale in scales else 1,
        1024 if scale in "mx" else 512 if scale in "l" else 1024,
    ) if scale else (depth, width, max_channels)

    gs = max(int(2 ** (len(ch) + 1)), 32)  # grid size (max stride)
    ch = [ch] if isinstance(ch, int) else ch
    nc = nc or 1

    save = []
    layers = []
    for i, (f, n_, m_, args) in enumerate(d["backbone"] + d["head"]):
        if m_ == "FSODDetect":
            m = FSODDetect
        elif m_ == "FSODDetect_ScaleReg" and FSODDetect_ScaleReg is not None:
            m = FSODDetect_ScaleReg
        else:
            m = (
                getattr(torch.nn, m_[3:])
                if "nn." in m_
                else getattr(torchvision.ops, m_[16:])
                if "torchvision.ops." in m_
                else globals().get(m_) or getattr(tasks_module, m_, None)
            )
            if m is None:
                raise KeyError(m_)

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n_ * depth), 1) if n_ > 1 else n_

        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in detect_like:
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m in {Segment, YOLOESegment, Segment26, YOLOESegment26}:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:
                m.legacy = legacy
        elif m is SemanticSegment:
            args.append([ch[x] for x in f])
        elif m is v10Detect:
            args.append([ch[x] for x in f])
        elif m is ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)

tasks_module.parse_model = _patched_parse_model

# ============================================================
# Step 2: Now import YOLO and run training
# ============================================================
from ultralytics import YOLO  # noqa: F401 - verify import works

# Import and run the training main
from scripts.train_fsod import main as train_main

if __name__ == "__main__":
    train_main()
