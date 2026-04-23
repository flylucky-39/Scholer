"""Direction C v1 evaluation: YOLO + CLIP verifier dual-path on CD-FSOD.

Pipeline (per image):
    1. ``model.predict(conf=conf, iou=iou, max_det=top_k)`` → post-NMS boxes
       in *original image coordinates* + per-box ``(conf, cls)``.
    2. CLIP verifier scores each surviving box against every class.
    3. ``fuse_box_scores`` produces a new per-box confidence.
    4. Class-aware NMS (torchvision ``batched_nms``) to clean up overlaps.
    5. Match against ground truth at IoU ∈ [0.5, 0.95] step 0.05.
    6. Accumulate and report mAP via ``ultralytics.utils.metrics.DetMetrics``.

Usage:
    python scripts/eval_fsod_dualpath.py \
        --config configs/cdfsod_DIOR_10shot.yaml \
        --weights runs/cdfsod_DIOR_10shot/cdfsod_DIOR_10shot_daf_a0.12/weights/best.pt \
        --text-proto runs/vlm_assets/dior_clip_vitb32_text_proto.pt \
        --fusion-mode fixed --gamma 0.5 \
        --top-k 100 --pad-ratio 1.2

Notes:
    * γ = 0 in ``fixed`` mode reproduces the YOLO-only baseline (sanity check).
    * ``--fusion-mode none`` skips the verifier entirely (pure baseline).
    * Latency is per-image wall time including verifier overhead.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401  (registers FSODDetect for checkpoint load)
from fsod.modules.dual_path_fusion import fuse_box_scores
from fsod.modules.vlm_verifier import CLIPVerifier


# ---------------------------------------------------------------------------
# CLI / config helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direction C dual-path eval.")
    parser.add_argument("--config", type=str, required=True,
                        help="CD-FSOD experiment yaml (e.g. configs/cdfsod_DIOR_10shot.yaml).")
    parser.add_argument("--weights", type=str, required=True,
                        help="Trained YOLO-FSOD checkpoint (.pt).")
    parser.add_argument("--text-proto", type=str, required=True,
                        help="CLIP text prototypes from extract_vlm_text_proto.py.")
    parser.add_argument("--fusion-mode", type=str, default="fixed",
                        choices=["none", "fixed", "multiplicative", "rerank"],
                        help="`none` = pure YOLO baseline (no verifier).")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Mixing weight for fixed mode.")
    parser.add_argument("--vlm-temperature", type=float, default=100.0)
    parser.add_argument("--vlm-only-below", type=float, default=1.0,
                        help="Only fuse VLM into boxes whose YOLO conf is BELOW this. "
                             "1.0 = always fuse (original behaviour); 0.5 = only "
                             "low-conf boxes get CLIP help, high-conf boxes keep YOLO score.")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Max detections per image after first NMS.")
    parser.add_argument("--pad-ratio", type=float, default=1.2,
                        help="Box context padding for CLIP crops.")
    parser.add_argument("--first-conf", type=float, default=0.001,
                        help="YOLO conf threshold (low → keep many candidates).")
    parser.add_argument("--first-iou", type=float, default=0.7,
                        help="YOLO NMS IoU threshold.")
    parser.add_argument("--final-iou", type=float, default=0.6,
                        help="Re-NMS IoU after fusion.")
    parser.add_argument("--final-conf", type=float, default=0.001,
                        help="Drop boxes whose fused confidence is below this.")
    parser.add_argument("--clip-model", type=str, default="",
                        help="Override CLIP HF id (default = stored in text-proto).")
    parser.add_argument("--clip-cache-dir", type=str, default="",
                        help="Optional HuggingFace cache dir for offline loading.")
    parser.add_argument("--clip-allow-online", action="store_true",
                        help="Allow downloading CLIP files if not cached locally.")
    parser.add_argument("--clip-input-size", type=int, default=224)
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Which split to evaluate on. Default `val` matches Phase-1 baseline.")
    parser.add_argument("--limit", type=int, default=0,
                        help="If > 0, only evaluate the first N images (smoke test).")
    parser.add_argument("--out-csv", type=str, default="runs/direction_c/dualpath_results.csv",
                        help="Append-mode CSV for the result row.")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Free-form tag stored in the CSV row.")
    return parser.parse_args()


def resolve_repo_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def find_test_manifest(output_root: Path) -> Path:
    """Locate the YOLO data yaml produced by ``prepare_cdfsod.py``."""
    candidates = list(output_root.glob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No yaml found in {output_root}")
    # Prefer one that has 'test' in name or is the only yaml.
    for c in candidates:
        if "test" in c.stem.lower() or "eval" in c.stem.lower():
            return c
    return candidates[0]


def load_image_list(data_yaml: Path, split: str = "val") -> list[Path]:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    field = cfg.get(split)
    if field is None:
        # Fallback to the other split if the requested one is missing.
        other = "test" if split == "val" else "val"
        field = cfg.get(other)
        if field is None:
            raise KeyError(f"data yaml {data_yaml} has no {split!r} or {other!r} entry")
        print(f"[warn] split={split!r} missing in {data_yaml.name}; falling back to {other!r}")
    p = Path(field).expanduser()
    if p.is_file() and p.suffix == ".txt":
        with p.open("r", encoding="utf-8") as f:
            return [Path(line.strip()) for line in f if line.strip()]
    if p.is_dir():
        return sorted(q for q in p.rglob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
    raise FileNotFoundError(f"split path not found: {p}")


def load_gt_for_image(image_path: Path, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Read YOLO-format label file paired with ``image_path``.

    Returns
    -------
    boxes_xyxy : np.ndarray  shape [G, 4] in pixel coords
    classes    : np.ndarray  shape [G] int
    """
    label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not label_path.exists():
        # Some manifests put labels alongside images
        label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    H, W = img.shape[:2]

    boxes: list[list[float]] = []
    cls_list: list[int] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
            x1 = (cx - w / 2.0) * W
            y1 = (cy - h / 2.0) * H
            x2 = (cx + w / 2.0) * W
            y2 = (cy + h / 2.0) * H
            boxes.append([x1, y1, x2, y2])
            cls_list.append(cls)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(cls_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Matching (standard YOLO algorithm)
# ---------------------------------------------------------------------------
def match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou_matrix: torch.Tensor,
    iouv: torch.Tensor,
) -> np.ndarray:
    """Return ``[num_pred, len(iouv)]`` boolean TP matrix."""
    n_pred = int(pred_classes.shape[0])
    n_iou = int(iouv.shape[0])
    tp = np.zeros((n_pred, n_iou), dtype=bool)
    if n_pred == 0 or true_classes.shape[0] == 0:
        return tp

    correct_class = (true_classes[:, None] == pred_classes[None, :]).cpu().numpy()
    iou = iou_matrix.cpu().numpy() * correct_class
    for i, t in enumerate(iouv.tolist()):
        m_indices = np.argwhere(iou >= t)  # [k, 2] (gt, pred)
        if m_indices.size == 0:
            continue
        ious_at_matches = iou[m_indices[:, 0], m_indices[:, 1]]
        order = ious_at_matches.argsort()[::-1]
        m_indices = m_indices[order]
        # Keep one match per gt
        _, gt_idx = np.unique(m_indices[:, 0], return_index=True)
        m_indices = m_indices[gt_idx]
        # Keep one match per pred
        _, pred_idx = np.unique(m_indices[:, 1], return_index=True)
        m_indices = m_indices[pred_idx]
        tp[m_indices[:, 1], i] = True
    return tp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    config = yaml.safe_load(resolve_repo_path(args.config).read_text(encoding="utf-8"))
    output_root = resolve_repo_path(config["output_root"])
    runs_dir = resolve_repo_path(config["runs_dir"])
    image_size = int(config["image_size"])
    weights_path = resolve_repo_path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    print(f"[setup] weights:    {weights_path}")
    print(f"[setup] device:     {device}")
    print(f"[setup] fusion:     {args.fusion_mode}  γ={args.gamma}  T={args.vlm_temperature}")
    print(f"[setup] top-k:      {args.top_k}  pad={args.pad_ratio}")

    model = YOLO(str(weights_path))
    yolo_names = model.names  # dict[int, str]
    num_classes = len(yolo_names)

    # Build verifier (skip if pure baseline).
    verifier: CLIPVerifier | None = None
    if args.fusion_mode != "none":
        verifier = CLIPVerifier(
            text_proto_path=resolve_repo_path(args.text_proto),
            model_name=args.clip_model or None,
            device=device,
            pad_ratio=args.pad_ratio,
            input_size=args.clip_input_size,
            batch_size=args.clip_batch_size,
            local_files_only=not args.clip_allow_online,
            cache_dir=(resolve_repo_path(args.clip_cache_dir) if args.clip_cache_dir else None),
        )
        verifier.assert_class_alignment(yolo_names)
        print(f"[setup] verifier:   CLIP backbone aligned to {len(verifier.class_names)} classes")

    data_yaml = find_test_manifest(output_root)
    image_paths = load_image_list(data_yaml, split=args.split)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    print(f"[setup] data yaml:  {data_yaml}")
    print(f"[setup] split:      {args.split}")
    print(f"[setup] images:     {len(image_paths)}")

    iouv = torch.linspace(0.5, 0.95, 10)

    all_tp: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_pred_cls: list[np.ndarray] = []
    all_target_cls: list[np.ndarray] = []
    latencies: list[float] = []

    for idx, img_path in enumerate(image_paths, start=1):
        if not img_path.exists():
            continue

        t0 = time.perf_counter()

        results = model.predict(
            source=str(img_path),
            conf=args.first_conf,
            iou=args.first_iou,
            max_det=args.top_k,
            imgsz=image_size,
            device=device,
            verbose=False,
        )
        if not results:
            latencies.append(time.perf_counter() - t0)
            continue
        det = results[0].boxes
        if det is None or det.shape[0] == 0:
            gt_boxes, gt_cls = load_gt_for_image(img_path, num_classes)
            if gt_cls.size > 0:
                all_target_cls.append(gt_cls)
            latencies.append(time.perf_counter() - t0)
            continue

        boxes_xyxy = det.xyxy.detach()                # [N, 4] in original image coords
        s_yolo = det.conf.detach().to(device)         # [N]
        yolo_cls = det.cls.detach().to(device).long() # [N]

        if verifier is not None and boxes_xyxy.shape[0] > 0:
            # Use the original-resolution image so crops match annotation coords.
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()  # uint8 CHW

            s_vlm = verifier.verify(img_t, boxes_xyxy)  # [N, C]
            new_conf = fuse_box_scores(
                s_yolo,
                yolo_cls,
                s_vlm,
                mode=args.fusion_mode,
                gamma=args.gamma,
                vlm_temperature=args.vlm_temperature,
            )
            # Selective fusion: high-confidence YOLO boxes bypass the verifier.
            if args.vlm_only_below < 1.0:
                bypass = s_yolo >= args.vlm_only_below
                if bypass.any():
                    new_conf = torch.where(bypass, s_yolo, new_conf)
        else:
            new_conf = s_yolo

        # Confidence filter.
        keep_mask = new_conf > args.final_conf
        boxes_xyxy = boxes_xyxy[keep_mask].to(device)
        new_conf = new_conf[keep_mask]
        yolo_cls = yolo_cls[keep_mask]

        # Re-NMS (class-aware).
        if boxes_xyxy.shape[0] > 0:
            keep = batched_nms(boxes_xyxy.float(), new_conf.float(), yolo_cls, args.final_iou)
            boxes_xyxy = boxes_xyxy[keep]
            new_conf = new_conf[keep]
            yolo_cls = yolo_cls[keep]

        latencies.append(time.perf_counter() - t0)

        # Match against GT.
        gt_boxes_np, gt_cls_np = load_gt_for_image(img_path, num_classes)
        if gt_cls_np.size > 0:
            all_target_cls.append(gt_cls_np)
        if boxes_xyxy.shape[0] == 0:
            continue
        if gt_boxes_np.size > 0:
            gt_boxes_t = torch.from_numpy(gt_boxes_np).to(device)
            gt_cls_t = torch.from_numpy(gt_cls_np).to(device)
            iou_matrix = box_iou(gt_boxes_t, boxes_xyxy)
            tp_image = match_predictions(yolo_cls, gt_cls_t, iou_matrix, iouv)
        else:
            tp_image = np.zeros((boxes_xyxy.shape[0], len(iouv)), dtype=bool)

        all_tp.append(tp_image)
        all_conf.append(new_conf.detach().cpu().numpy())
        all_pred_cls.append(yolo_cls.detach().cpu().numpy())

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"  [{idx}/{len(image_paths)}] avg latency = "
                  f"{1000.0 * np.mean(latencies):.1f} ms")

    if not all_tp:
        raise RuntimeError("No detections produced. Check weights / data path.")

    tp = np.concatenate(all_tp, axis=0)
    conf_arr = np.concatenate(all_conf, axis=0)
    pred_cls_arr = np.concatenate(all_pred_cls, axis=0)
    target_cls_arr = np.concatenate(all_target_cls, axis=0) if all_target_cls else np.zeros((0,), dtype=np.int64)

    save_dir = runs_dir / "dualpath_eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = DetMetrics(save_dir=save_dir, names=yolo_names)
    metrics.process(tp, conf_arr, pred_cls_arr, target_cls_arr)

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    map75 = float(metrics.box.map75)
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    print("\n" + "=" * 70)
    print(f" fusion_mode={args.fusion_mode}  gamma={args.gamma}  T={args.vlm_temperature}")
    print(f" mAP50      : {map50:.4f}")
    print(f" mAP50-95   : {map50_95:.4f}")
    print(f" mAP75      : {map75:.4f}")
    print(f" P / R      : {mp:.4f} / {mr:.4f}")
    print(f" latency    : {1000.0 * mean_latency:.2f} ms / image")
    print("=" * 70)

    # Append a row to CSV log.
    out_csv = resolve_repo_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    row = {
        "run_tag": args.run_tag,
        "weights": str(weights_path),
        "split": args.split,
        "fusion_mode": args.fusion_mode,
        "gamma": args.gamma,
        "vlm_temperature": args.vlm_temperature,
        "vlm_only_below": args.vlm_only_below,
        "top_k": args.top_k,
        "pad_ratio": args.pad_ratio,
        "first_conf": args.first_conf,
        "first_iou": args.first_iou,
        "final_iou": args.final_iou,
        "n_images": len(image_paths),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "mAP75": round(map75, 4),
        "P": round(mp, 4),
        "R": round(mr, 4),
        "latency_ms": round(1000.0 * mean_latency, 2),
    }
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n[csv] appended: {out_csv}")

    # Also dump a per-class json snapshot next to the csv.
    snapshot = {
        "row": row,
        "per_class_ap50": [float(x) for x in metrics.box.ap50.tolist()] if hasattr(metrics.box, "ap50") else [],
        "per_class_ap": [float(x) for x in metrics.box.ap.tolist()] if hasattr(metrics.box, "ap") else [],
        "class_names": [yolo_names[i] for i in sorted(yolo_names)],
    }
    snapshot_path = save_dir / f"{args.run_tag or 'run'}_{args.fusion_mode}_g{args.gamma}.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[json] {snapshot_path}")


if __name__ == "__main__":
    main()
