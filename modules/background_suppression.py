"""Background Suppression for anchor-free few-shot object detection.

Anchor-free detectors (YOLO) predict classification scores at every spatial
location.  In few-shot regimes, class prototypes may capture generic background
textures (edges, corners, repeated patterns), causing false positives on
background regions.  This module provides two complementary mechanisms:

1. Prototype Orthogonalisation (pre-init):
   Extract a "background prototype" from non-object regions in base_train
   images, then orthogonalise each class prototype against it:
       p' = normalize(p - beta * (p · b̂) * b̂)
   This removes the background-aligned component from every class prototype.

2. Inference-time Background Suppression (CosineConv2d forward):
   For each spatial location, subtract the cosine similarity to the
   background prototype from all class scores:
       score = s * (cos(x, w) - gamma * cos(x, b)) + bias
   This penalises background-like features at inference time (zero extra
   parameters — gamma is a fixed scalar).

Usage:
    from fsod.modules.background_suppression import (
        extract_background_prototype,
        refine_prototypes_with_bg_suppression,
    )

    bg_proto = extract_background_prototype(base_weights, data_root, all_classes)
    refined = refine_prototypes_with_bg_suppression(prototypes, bg_proto, beta=0.5)
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from ultralytics import YOLO

from fsod.modules.prototype import _extract_cv3_features, _load_labels, _cxcywh_to_xyxy


@torch.no_grad()
def extract_background_prototype(
    base_weights: str | Path,
    data_root: str | Path,
    all_classes: list[str],
    imgsz: int = 640,
    device: str = "cuda:0",
    max_images: int = 200,
    max_per_image: int = 20,
    seed: int = 42,
) -> torch.Tensor:
    """Extract a background prototype from non-object regions in base_train images.

    For each image, runs a forward pass through backbone + neck + cv3 penultimate
    layer, then samples feature vectors from spatial locations that do NOT overlap
    with any ground-truth box.  The average of all sampled features gives a
    "background prototype" — the typical feature vector produced by background
    regions.

    Args:
        base_weights: Path to base-pretrained YOLO checkpoint.
        data_root: Dataset root (must contain manifests/base_train.txt).
        all_classes: List of all class names (for label index mapping).
        imgsz: Image size for inference.
        device: Torch device string.
        max_images: Max base_train images to process (subsample for speed).
        max_per_image: Max background features to sample per scale per image.
        seed: Random seed for subsampling.

    Returns:
        Background prototype tensor of shape (c3,) on CPU, where c3 is the
        cv3 penultimate channel count (256 for yolo11s).
    """
    data_root = Path(data_root)
    base_weights = Path(base_weights)
    manifest_path = data_root / "manifests" / "base_train.txt"

    if not manifest_path.exists():
        print(f"  WARNING: base_train manifest not found at {manifest_path}. "
              f"Returning zero background prototype.")
        return torch.zeros(256)

    image_paths = [
        Path(line.strip())
        for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()
    ]

    if len(image_paths) > max_images:
        rng = random.Random(seed)
        image_paths = rng.sample(image_paths, max_images)

    print(f"  Extracting background prototype from {len(image_paths)} base_train images...")

    # Load model
    model = YOLO(str(base_weights))
    model.to(device)
    model.model.eval()

    cls_idx_to_name = {i: name for i, name in enumerate(all_classes)}
    all_bg_features: list[torch.Tensor] = []

    for img_idx, img_path in enumerate(image_paths):
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        orig_w, orig_h = img_pil.size
        img_resized = img_pil.resize((imgsz, imgsz))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Load labels
        label_path = data_root / "labels" / "base_train" / (img_path.stem + ".txt")
        labels = _load_labels(label_path)
        if not labels:
            continue

        # Convert GT boxes to resized image coordinates
        scale_x = imgsz / orig_w
        scale_y = imgsz / orig_h
        gt_boxes_abs = []
        for cls_id, cx, cy, w, h in labels:
            x1 = (cx - w / 2) * orig_w * scale_x
            y1 = (cy - h / 2) * orig_h * scale_y
            x2 = (cx + w / 2) * orig_w * scale_x
            y2 = (cy + h / 2) * orig_h * scale_y
            gt_boxes_abs.append((x1, y1, x2, y2))

        # Extract multi-scale penultimate cv3 features
        scale_features = _extract_cv3_features(model, img_tensor)

        # For each scale, sample non-object spatial locations
        for scale_idx, feat_map in enumerate(scale_features):
            _, c, h, w = feat_map.shape
            stride = imgsz / h

            # Build foreground mask: 1 = background, 0 = object region
            bg_mask = torch.ones(h, w, device=device)
            for gx1, gy1, gx2, gy2 in gt_boxes_abs:
                fx1 = max(0, int(gx1 / stride))
                fy1 = max(0, int(gy1 / stride))
                fx2 = min(w, int(gx2 / stride + 1))
                fy2 = min(h, int(gy2 / stride + 1))
                bg_mask[fy1:fy2, fx1:fx2] = 0

            # Collect background features
            bg_indices = bg_mask.nonzero(as_tuple=False)
            if bg_indices.numel() == 0:
                continue
            if len(bg_indices) > max_per_image:
                perm = torch.randperm(len(bg_indices), device=device)[:max_per_image]
                bg_indices = bg_indices[perm]

            for yi, xi in bg_indices:
                all_bg_features.append(feat_map[0, :, yi.item(), xi.item()])

        if (img_idx + 1) % 50 == 0:
            print(f"    Processed {img_idx + 1}/{len(image_paths)} images, "
                  f"{len(all_bg_features)} background features collected")

    del model
    torch.cuda.empty_cache()

    if not all_bg_features:
        print("  WARNING: No background features collected. Returning zero vector.")
        return torch.zeros(256)

    stacked = torch.stack(all_bg_features)  # (N, c3)
    bg_proto = stacked.mean(dim=0).cpu()
    print(f"  Background prototype: {len(all_bg_features)} features from "
          f"{len(image_paths)} images, norm={bg_proto.norm():.3f}")
    return bg_proto


def refine_prototypes_with_bg_suppression(
    prototypes: dict[str, torch.Tensor],
    bg_proto: torch.Tensor,
    beta: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Orthogonalise class prototypes against the background prototype.

    For each class prototype p, subtracts the component that aligns with the
    background direction b̂:

        p' = normalize( p - beta * <p, b̂> * b̂ )

    where b̂ = bg_proto / ||bg_proto||.

    This removes the background-aligned feature component from each class
    prototype, reducing false activations on background regions.

    Args:
        prototypes: Dict class_name → prototype tensor (c3,).
        bg_proto: Background prototype tensor (c3,).
        beta: Suppression strength (0 = no suppression, 1 = full orthogonalisation).

    Returns:
        Dict class_name → refined prototype tensor (c3,), same device as input.
    """
    if bg_proto.norm() == 0:
        print("  Background prototype is zero — skipping refinement.")
        return prototypes

    bg_norm = F.normalize(bg_proto, dim=0)
    device = next(iter(prototypes.values())).device
    bg_norm = bg_norm.to(device)

    refined: dict[str, torch.Tensor] = {}
    for cls_name, proto in prototypes.items():
        if proto.norm() == 0:
            refined[cls_name] = proto
            continue

        # Projection of proto onto background direction
        proj = torch.dot(proto, bg_norm) * bg_norm
        # Subtract projection, then normalize
        refined_proto = proto - beta * proj
        refined[cls_name] = F.normalize(refined_proto, dim=0)

    # Report statistics
    cos_before = []
    cos_after = []
    for cls_name in prototypes:
        if prototypes[cls_name].norm() == 0:
            continue
        cos_before.append(
            torch.dot(F.normalize(prototypes[cls_name], dim=0), bg_norm).item()
        )
        cos_after.append(
            torch.dot(refined[cls_name], bg_norm).item()
        )
    if cos_before:
        print(f"  Prototype refinement (beta={beta:.2f}): "
              f"cos_bg {np.mean(cos_before):.4f} → {np.mean(cos_after):.4f} "
              f"(avg reduction: {np.mean(cos_before) - np.mean(cos_after):.4f})")

    return refined