"""扫 confidence 阈值，找最佳 P/R/mAP 平衡点。

Usage:
  # Novel-only eval (推荐)
  python scripts/sweep_confidence.py \
      --weights runs/fsod_baseline/novel_finetune_cosine_fused/weights/best.pt \
      --config configs/baseline_voc_10shot.yaml --novel-only

  # All 20 classes
  python scripts/sweep_confidence.py \
      --weights runs/fsod_baseline/novel_finetune_cosine_fused/weights/best.pt \
      --config configs/baseline_voc_10shot.yaml
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import fsod.modules  # noqa: F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confidence threshold sweep for FSOD eval.")
    p.add_argument("--config", required=True, help="Experiment config yaml.")
    p.add_argument("--weights", required=True, help="Path to best.pt.")
    p.add_argument("--thresholds", default="0.001,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5",
                   help="Comma-separated confidence thresholds.")
    p.add_argument("--novel-only", action="store_true",
                   help="Only report novel class metrics (ignores base classes).")
    return p.parse_args()


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(resolve_repo_path(args.config)).open(encoding="utf-8"))
    output_root = resolve_repo_path(config["output_root"])
    weights_path = resolve_repo_path(args.weights)
    novel_classes = set(config["novel_classes"])

    data_yaml = output_root / "voc_fsod_eval_all.yaml"
    if not data_yaml.exists():
        data_yaml = output_root / "voc_fsod_finetune.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No eval yaml in {output_root}")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    mode = "novel-only" if args.novel_only else "all 20 classes"
    print(f"Weights: {weights_path}")
    print(f"Data: {data_yaml}")
    print(f"Mode: {mode}")
    print(f"Thresholds: {thresholds}")
    print()

    results = []
    for conf in thresholds:
        model = YOLO(str(weights_path))
        metrics = model.val(
            data=str(data_yaml), split="test",
            imgsz=int(config["image_size"]),
            device=config["device"],
            conf=conf,
            verbose=False,
            plots=False,
        )

        # Get per-class AP and class indices
        ap50 = np.array(metrics.box.ap50) if hasattr(metrics.box, "ap50") else np.array([])
        ap = np.array(metrics.box.ap) if hasattr(metrics.box, "ap") else np.array([])
        class_names = list(metrics.names.values()) if hasattr(metrics, "names") else []

        if args.novel_only and len(ap50) > 0 and class_names:
            novel_mask = np.array([name in novel_classes for name in class_names])
            novel_ap50 = ap50[novel_mask[:len(ap50)]] if len(ap50) >= len(novel_mask) else ap50
            novel_ap = ap[novel_mask[:len(ap)]] if len(ap) >= len(novel_mask) else ap
            m50 = float(np.mean(novel_ap50)) if len(novel_ap50) > 0 else 0.0
            m5095 = float(np.mean(novel_ap)) if len(novel_ap) > 0 else 0.0
        else:
            m50 = float(metrics.box.map50)
            m5095 = float(metrics.box.map)

        mp = float(metrics.box.mp)
        mr = float(metrics.box.mr)
        m75 = float(metrics.box.map75)

        results.append((conf, mp, mr, m50, m5095, m75))
        print(f"  conf={conf:<5} P={mp:.4f}  R={mr:.4f}  "
              f"mAP50={m50:.4f}  mAP50-95={m5095:.4f}  mAP75={m75:.4f}")

    # Summary
    print("\n" + "=" * 75)
    print(f"{'conf':>6}  {'P':>8}  {'R':>8}  {'mAP50':>8}  {'mAP50-95':>8}  {'mAP75':>8}")
    print("-" * 75)
    best_map50 = max(results, key=lambda r: r[3])
    best_map = max(results, key=lambda r: r[4])
    for conf, mp, mr, m50, m5095, m75 in results:
        flag = ""
        if (conf, mp, mr, m50, m5095, m75) == best_map50:
            flag += " ← best mAP50"
        if (conf, mp, mr, m50, m5095, m75) == best_map:
            flag += " ← best mAP50-95"
        print(f"  {conf:>4.2f}  {mp:>8.4f}  {mr:>8.4f}  "
              f"{m50:>8.4f}  {m5095:>8.4f}  {m75:>8.4f}{flag}")

    # Recommendation
    print(f"\n🔍 分析（{mode}）：")
    for conf, mp, mr, m50, m5095, _ in results:
        if mp > 0.2:
            print(f"  conf={conf:.2f}: P={mp:.3f} 可接受，mAP50-95={m5095:.3f}")
            break
    if not any(r[1] > 0.2 for r in results):
        print(f"  ⚠️ 最高 P={max(r[1] for r in results):.3f}。提高阈值无法有效压 FP。")
        print(f"  建议：提高 initial temperature (15-20) 重训 + per-class bias。")

    # Save
    out = {
        "weights": str(weights_path),
        "mode": mode,
        "thresholds": [{"conf": r[0], "P": r[1], "R": r[2],
                        "mAP50": r[3], "mAP50-95": r[4], "mAP75": r[5]}
                       for r in results],
    }
    out_path = weights_path.parent.parent / "conf_sweep.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"\n结果保存: {out_path}")


if __name__ == "__main__":
    main()
