"""快速验证：相同best.pt，split=val vs split=test 的结果差异"""
import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path('/root/epfs/07_FSOD_LLM/fsod')
sys.path.insert(0, str(PROJECT_ROOT))
import fsod.modules  # noqa: F401
from ultralytics import YOLO


def patch_cosine_buffers(model):
    sd = model.state_dict()
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'CosineConv2d':
            for buf_name in ['weight_prior', 'weight_prior_mask', 'fixed_blend_alpha']:
                key = f'{name}.{buf_name}'
                if key not in sd:
                    if buf_name == 'weight_prior_mask':
                        buf = torch.zeros(module.out_channels, dtype=torch.bool)
                    elif buf_name == 'fixed_blend_alpha':
                        buf = torch.tensor(0.5)
                    else:
                        buf = torch.zeros(module.out_channels, module.in_channels)
                    module.register_buffer(buf_name, buf)
                    sd[key] = buf
    model.load_state_dict(sd, strict=False)
    return model


# Test best.pt
model = YOLO('runs/voc_fsod_10shot/novel_finetune_cosine_fused/weights/best.pt')
patch_cosine_buffers(model.model)

for split in ['val', 'test']:
    m = model.val(
        data='data/voc_fsod_split1_10shot/voc_fsod_finetune.yaml',
        split=split, imgsz=640, device=0, conf=0.001,
        verbose=False, plots=False,
    )
    print(f'best.pt | split={split}: mAP50={m.box.map50:.4f}, mAP50-95={m.box.map:.4f}, P={m.box.mp:.4f}, R={m.box.mr:.4f}')
    # Per-class
    names = list(m.names.values())
    print(f'  Per-class mAP50:')
    for i, name in enumerate(names):
        if i < len(m.box.ap50):
            print(f'    {name}: {m.box.ap50[i]:.3f}')
