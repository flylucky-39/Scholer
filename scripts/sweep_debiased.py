#!/usr/bin/env python3
"""Confidence sweep on old checkpoints with debiased inference (CosineConv2d bias subtract)."""
import sys, json, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "ultralytics"))
import fsod.modules  # noqa
from ultralytics import YOLO

CONF_THRS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
MODELS = {
    "1-shot": ("runs/fsod_1shot/novel_finetune_cosine_proto/weights/best.pt",
               "configs/baseline_voc_1shot.yaml"),
    "3-shot": ("runs/fsod_3shot/novel_finetune_cosine_proto/weights/best.pt",
               "configs/baseline_voc_3shot.yaml"),
    "5-shot": ("runs/fsod_5shot/novel_finetune_cosine_proto/weights/best.pt",
               "configs/baseline_voc_5shot.yaml"),
    "10-shot": ("runs/voc_fsod_10shot/novel_finetune_cosine_proto/weights/best.pt",
                "configs/baseline_voc_10shot.yaml"),
}

print("=" * 82)
print("  Confidence Sweep — 旧模型 + 新 debiased 推理")
print("  (训练不变，推理时减去 CosineConv2d.bias)")
print("=" * 82)

for shot, (weights_path, config_path) in MODELS.items():
    p = Path(weights_path)
    if not p.exists():
        print(f"\n  SKIP {shot}: {p} not found")
        continue

    cfg = yaml.safe_load(open(config_path))
    data_yaml = Path(cfg["output_root"])
    if not data_yaml.is_absolute():
        data_yaml = Path.cwd() / data_yaml
    data_yaml = data_yaml / "voc_fsod_eval_all.yaml"

    print(f"\n── {shot} ──")
    print(f"  {'conf':>7}  {'P':>8}  {'R':>8}  {'mAP50':>8}  {'mAP50-95':>8}")
    print(f"  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    results = []
    for conf in CONF_THRS:
        model = YOLO(str(p))
        metrics = model.val(
            data=str(data_yaml), split="test",
            imgsz=640, device=0, conf=conf,
            verbose=False, plots=False,
        )
        mp = float(metrics.box.mp)
        mr = float(metrics.box.mr)
        m50 = float(metrics.box.map50)
        m5095 = float(metrics.box.map)
        results.append((conf, mp, mr, m50, m5095))

    best = max(results, key=lambda r: r[3])
    for conf, mp, mr, m50, m5095 in results:
        flag = " ← best" if conf == best[0] else ""
        print(f"  {conf:>7.3f}  {mp:>8.4f}  {mr:>8.4f}  {m50:>8.4f}  {m5095:>8.4f}{flag}")

    # Save
    out_dir = p.parent.parent
    out = {
        "weights": str(p),
        "mode": "novel-only-debiased",
        "thresholds": [
            {"conf": c, "P": mp, "R": mr, "mAP50": m50, "mAP50-95": m5095}
            for c, mp, mr, m50, m5095 in results
        ],
    }
    json.dump(out, open(out_dir / "conf_sweep_debiased.json", "w"), indent=2)

print("\n✅ Done!")
