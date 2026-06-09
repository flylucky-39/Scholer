"""Prototype extraction for few-shot cosine classifier initialization.

Given a base-pretrained YOLO model and a few-shot support set (novel_finetune),
extract per-class mean feature vectors (prototypes) from the cv3 classification
branch.  These prototypes are used to initialize CosineConv2d weights, giving
the cosine classifier a much better starting point than random initialization.

Pipeline:
  1. Load base-pretrained YOLO model
  2. For each support image, run a forward pass through backbone + neck + cv3
     (up to the penultimate layer, i.e., the 256-dim feature before final 1x1)
  3. Use GT bboxes to RoIAlign crop features at each scale
  4. Average all crops per class → 256-dim prototype per class
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.ops import roi_align
from pathlib import Path
from PIL import Image
import numpy as np

from ultralytics import YOLO


def _load_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO-format labels: class_id cx cy w h (normalized)."""
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        labels.append((cls_id, cx, cy, w, h))
    return labels


def _cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float,
                     img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert normalized cxcywh to absolute xyxy."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def _extract_cv3_features(model: YOLO, img_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Run forward pass and extract features at the penultimate layer of each cv3 branch.

    Uses the same forward logic as ultralytics _predict_once, but stops at the
    detect head and manually runs cv3[i][:-1] to get penultimate features.

    Returns a list of 3 tensors, one per scale:
      - scale 0 (P3/8):  shape (1, c3, H/8, W/8)
      - scale 1 (P4/16): shape (1, c3, H/16, W/16)
      - scale 2 (P5/32): shape (1, c3, H/32, W/32)
    """
    net = model.model  # DetectionModel
    detect = net.model[-1]  # Detect head (last module)

    # Run backbone + neck using ultralytics' own forward logic
    y = []  # intermediate outputs
    x = img_tensor
    for m in net.model:
        if m is detect:
            break
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in net.save else None)

    # Get detect head inputs (from indices in detect.f)
    if isinstance(detect.f, list):
        head_inputs = [y[j] for j in detect.f]
    else:
        head_inputs = [x]

    # Extract penultimate cv3 features for each scale
    features = []
    for i, xi in enumerate(head_inputs):
        feat = xi
        cv3_seq = detect.cv3[i]
        # Run all layers except the last (CosineConv2d / Conv2d)
        for layer in list(cv3_seq)[:-1]:
            feat = layer(feat)
        features.append(feat)

    return features


@torch.no_grad()
def extract_prototypes(
    base_weights: str | Path,
    data_root: str | Path,
    novel_classes: list[str],
    all_classes: list[str],
    imgsz: int = 640,
    device: str = "cuda:0",
    augment: bool = False,
    augment_k: int = 4,
) -> dict[str, torch.Tensor]:
    """Extract per-class prototypes from support set.

    Args:
        base_weights: Path to base-pretrained YOLO checkpoint.
        data_root: Path to data root (e.g., data/voc_fsod_split1_10shot).
        novel_classes: List of novel class names.
        all_classes: List of all class names (for index mapping).
        imgsz: Image size for inference.
        device: Device string.
        augment: Whether to apply data augmentation to support images.
        augment_k: Number of augmented variants per image (used only if augment=True).

    Returns:
        Dict mapping class_name → prototype tensor of shape (c3,) where c3=256 for yolo11s.
    """
    data_root = Path(data_root)
    base_weights = Path(base_weights)

    # Load base model
    model = YOLO(str(base_weights))
    model.to(device)
    model.model.eval()

    # Build class index → name mapping
    cls_idx_to_name = {i: name for i, name in enumerate(all_classes)}
    novel_set = set(novel_classes)

    # Read support manifest
    manifest_path = data_root / "manifests" / "novel_finetune.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    image_paths = [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()]

    # Collect features per class
    class_features: dict[str, list[torch.Tensor]] = {name: [] for name in novel_classes}

    # Define augmentation transforms
    if augment:
        augmentations = [
            T.RandomHorizontalFlip(p=1.0),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        ]
        augmentations = augmentations[:augment_k]
        print(f"  Augmentation enabled: {len(augmentations)} variants per image")

    for img_path in image_paths:
        # Load and preprocess image
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Generate image variants (original + augmented)
        variants = [img_pil]
        if augment:
            for aug in augmentations:
                variants.append(aug(img_pil))

        for variant in variants:
            # Resize to imgsz (letterbox-style: resize to fit, then pad)
            # For simplicity, use direct resize (matches training augmentation at eval)
            img_resized = variant.resize((imgsz, imgsz))
            img_np = np.array(img_resized, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

            # Load labels
            label_path = data_root / "labels" / "novel_finetune" / (img_path.stem + ".txt")
            labels = _load_labels(label_path)
            if not labels:
                continue

            # Extract multi-scale cv3 features
            scale_features = _extract_cv3_features(model, img_tensor)

            # For each GT box, RoIAlign crop from the best-matching scale
            for cls_id, cx, cy, w, h in labels:
                cls_name = cls_idx_to_name.get(cls_id)
                if cls_name is None or cls_name not in novel_set:
                    continue

                x1, y1, x2, y2 = _cxcywh_to_xyxy(cx, cy, w, h, imgsz, imgsz)
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w < 2 or box_h < 2:
                    continue

                # Pick scale based on box size (small→P3, medium→P4, large→P5)
                box_area = box_w * box_h
                if box_area < 96 * 96:
                    scale_idx = 0  # P3/8
                elif box_area < 192 * 192:
                    scale_idx = 1  # P4/16
                else:
                    scale_idx = 2  # P5/32

                feat_map = scale_features[scale_idx]  # (1, c3, H_s, W_s)
                stride = imgsz / feat_map.shape[-1]  # 8, 16, or 32

                # Scale bbox to feature map coordinates
                rois = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32, device=device)

                # RoIAlign: output 1x1 spatial → (1, c3, 1, 1)
                pooled = roi_align(feat_map, rois, output_size=(1, 1), spatial_scale=1.0 / stride)
                feat_vec = pooled.squeeze()  # (c3,)
                class_features[cls_name].append(feat_vec)

    # Average features per class → prototypes
    prototypes = {}
    n_images = len(image_paths)
    for cls_name in novel_classes:
        feats = class_features[cls_name]
        if not feats:
            print(f"WARNING: No features extracted for class '{cls_name}'. Using zero vector.")
            # Infer c3 from any available feature
            c3 = next((f.shape[0] for v in class_features.values() for f in v), 256)
            prototypes[cls_name] = torch.zeros(c3, device=device)
        else:
            stacked = torch.stack(feats)  # (N, c3)
            prototypes[cls_name] = stacked.mean(dim=0)  # (c3,)
            print(f"  {cls_name}: {len(feats)} features ({n_images} images × {len(feats)//max(n_images,1)} variants) → prototype norm={prototypes[cls_name].norm():.3f}")

    return prototypes


def init_cosine_head_with_prototypes(
    model: YOLO,
    prototypes: dict[str, torch.Tensor],
    novel_classes: list[str],
    all_classes: list[str],
) -> None:
    """Initialize CosineConv2d weights for novel classes using prototypes.

    For novel classes: set weight row = L2-normalized prototype.
    For base classes: keep the existing (transferred from base pretrain or random) weights.

    Args:
        model: YOLO model with FSODDetect head (already has weights loaded).
        prototypes: Dict class_name → prototype vector (c3,).
        novel_classes: List of novel class names.
        all_classes: List of all class names.
    """
    cls_name_to_idx = {name: i for i, name in enumerate(all_classes)}
    detect = model.model.model[-1]

    for i in range(detect.nl):
        cosine_layer = detect.cv3[i][-1]  # CosineConv2d
        weight = cosine_layer.weight.data  # (nc, c3)

        for cls_name in novel_classes:
            if cls_name not in prototypes:
                continue
            idx = cls_name_to_idx.get(cls_name)
            if idx is None:
                continue
            proto = prototypes[cls_name].to(weight.device)
            # L2-normalize before assigning (CosineConv2d normalizes in forward anyway)
            weight[idx] = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)

        print(f"  Scale {i}: initialized {len(novel_classes)} novel class weights with prototypes")
