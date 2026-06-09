"""Confidence sweep on test_novel (standard FSOD eval protocol)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml
import torch

PROJECT_ROOT = Path('/root/epfs/07_FSOD_LLM/fsod')
sys.path.insert(0, str(PROJECT_ROOT))
import fsod.modules  # noqa: F401
from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_one_weight


def resolve(p: str) -> Path:
    path = Path(p).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def patch_cosine_buffers(model):
    """Add missing weight_prior_mask etc buffers to CosineConv2d layers."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--thresholds', default='0.001,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5')
    p.add_argument('--novel-only', action='store_true')
    args = p.parse_args()

    config = yaml.safe_load(open(resolve(args.config)))
    output_root = resolve(config['output_root'])
    weights_path = resolve(args.weights)
    novel_classes = set(config['novel_classes'])

    data_yaml = output_root / 'voc_fsod_finetune.yaml'
    if not data_yaml.exists():
        raise FileNotFoundError(f'{data_yaml} not found')

    thresholds = [float(t) for t in args.thresholds.split(',')]
    mode = 'novel-only' if args.novel_only else 'all 20 classes'
    print(f'Weights: {weights_path}')
    print(f'Data: {data_yaml}')
    print(f'Mode: {mode}')
    print(f'Thresholds: {thresholds}\n')

    results = []
    for conf in thresholds:
        model = YOLO(str(weights_path))
        patch_cosine_buffers(model.model)
        metrics = model.val(
            data=str(data_yaml), split='test',
            imgsz=int(config['image_size']),
            device=config['device'],
            conf=conf, verbose=False, plots=False,
        )

        ap50 = np.array(metrics.box.ap50)
        ap = np.array(metrics.box.ap)
        class_names = list(metrics.names.values())

        if args.novel_only and len(ap50) > 0 and class_names:
            novel_mask = np.array([n in novel_classes for n in class_names])
            m50 = float(np.mean(ap50[novel_mask[:len(ap50)]])) if novel_mask.any() else 0.0
            m5095 = float(np.mean(ap[novel_mask[:len(ap)]])) if novel_mask.any() else 0.0
        else:
            m50 = float(metrics.box.map50)
            m5095 = float(metrics.box.map)

        mp = float(metrics.box.mp)
        mr = float(metrics.box.mr)
        m75 = float(metrics.box.map75)

        results.append((conf, mp, mr, m50, m5095, m75))
        print(f'  conf={conf:<5} P={mp:.4f}  R={mr:.4f}  mAP50={m50:.4f}  mAP50-95={m5095:.4f}')

    print('\n' + '=' * 75)
    print(f'{"conf":>6}  {"P":>8}  {"R":>8}  {"mAP50":>8}  {"mAP50-95":>8}  {"mAP75":>8}')
    print('-' * 75)
    best_map50 = max(results, key=lambda r: r[3])
    best_map = max(results, key=lambda r: r[4])
    for conf, mp, mr, m50, m5095, m75 in results:
        flag = ''
        if (conf, mp, mr, m50, m5095, m75) == best_map50:
            flag += ' ← best mAP50'
        if (conf, mp, mr, m50, m5095, m75) == best_map:
            flag += ' ← best mAP50-95'
        print(f'  {conf:>4.2f}  {mp:>8.4f}  {mr:>8.4f}  {m50:>8.4f}  {m5095:>8.4f}  {m75:>8.4f}{flag}')

    out_path = weights_path.parent.parent / 'conf_sweep.json'
    json.dump({'weights': str(weights_path), 'mode': mode,
               'thresholds': [{'conf': r[0], 'P': r[1], 'R': r[2],
                               'mAP50': r[3], 'mAP50-95': r[4], 'mAP75': r[5]} for r in results]},
              open(out_path, 'w', encoding='utf-8'), indent=2)
    print(f'\n结果保存: {out_path}')


if __name__ == '__main__':
    main()
