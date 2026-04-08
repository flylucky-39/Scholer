"""Adaptation layer: Florence-2 text encoder → MLP → CosineConv2d weight initialization.

Maps Florence-2 class name embeddings (1024-dim) to YOLO visual feature space (c3-dim)
using a lightweight MLP trained on base class (text_embedding, visual_prototype) pairs.

Pipeline:
  1. Extract Florence-2 text encoder embeddings for all 20 VOC class names (1024-dim)
  2. Extract visual prototypes for 15 base classes from base pretrain model (c3-dim)
  3. Train MLP on base class pairs: MLP(text_emb) ≈ visual_proto
  4. Apply trained MLP to novel class text embeddings → CosineConv2d weight init

At inference time: zero extra cost — Florence-2 is only used during weight initialization.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.ops import roi_align

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 1. Florence-2 Text Embedding Extraction
# ---------------------------------------------------------------------------

def extract_florence2_text_embeddings(
    model_path: str | Path,
    class_names: list[str],
    device: str = "cuda:0",
) -> dict[str, torch.Tensor]:
    """Extract text embeddings for class names from Florence-2 text encoder.

    Uses the BART encoder inside Florence-2 to produce contextual embeddings,
    then mean-pools over token positions to get a fixed-size vector per class.

    Args:
        model_path: Path to local Florence-2 model directory.
        class_names: Class name strings to embed.
        device: Torch device.

    Returns:
        Dict mapping class_name → tensor of shape (1024,) on CPU.
    """
    from fsod.florence2 import resolve_local_model_dir
    from transformers import AutoModelForCausalLM, AutoProcessor

    resolved_path = str(resolve_local_model_dir(model_path))

    processor = AutoProcessor.from_pretrained(
        resolved_path, trust_remote_code=True, local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    encoder = model.language_model.get_encoder()

    embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in class_names:
            tokens = processor.tokenizer(
                name, return_tensors="pt", padding=False,
            ).to(device)
            enc_out = encoder(input_ids=tokens["input_ids"], return_dict=True)
            # Mean pool over sequence dim → (1024,)
            emb = enc_out.last_hidden_state.mean(dim=1).squeeze(0)
            embeddings[name] = emb.cpu()

    del model, encoder, processor
    torch.cuda.empty_cache()

    print(f"  Extracted {len(embeddings)} Florence-2 text embeddings (dim={next(iter(embeddings.values())).shape[0]})")
    return embeddings


# ---------------------------------------------------------------------------
# 2. Base Class Visual Prototype Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_base_prototypes(
    base_weights: str | Path,
    data_root: str | Path,
    base_classes: list[str],
    all_classes: list[str],
    imgsz: int = 640,
    device: str = "cuda:0",
    max_images: int = 500,
    seed: int = 3407,
) -> dict[str, torch.Tensor]:
    """Extract per-class visual prototypes for base classes from base_train data.

    Same RoIAlign approach as prototype.extract_prototypes, but operates on
    the base_train split (many images, base classes only).

    Args:
        base_weights: Path to base-pretrained YOLO checkpoint.
        data_root: Dataset root (e.g., data/voc_fsod_split1_10shot).
        base_classes: List of base class names (15 for VOC).
        all_classes: List of all class names (20 for VOC, for index mapping).
        imgsz: Image size for inference.
        device: Torch device.
        max_images: Max images to process (subsample for speed).
        seed: Random seed for subsampling.

    Returns:
        Dict mapping class_name → prototype tensor of shape (c3,) on CPU.
    """
    from fsod.modules.prototype import _extract_cv3_features, _load_labels, _cxcywh_to_xyxy

    data_root = Path(data_root)
    manifest_path = data_root / "manifests" / "base_train.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Base train manifest not found: {manifest_path}")

    image_paths = [
        Path(line.strip())
        for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()
    ]

    if len(image_paths) > max_images:
        rng = random.Random(seed)
        image_paths = rng.sample(image_paths, max_images)

    print(f"  Extracting base prototypes from {len(image_paths)} images...")

    model = YOLO(str(base_weights))
    model.to(device)
    model.model.eval()

    cls_idx_to_name = {i: name for i, name in enumerate(all_classes)}
    base_set = set(base_classes)
    class_features: dict[str, list[torch.Tensor]] = {name: [] for name in base_classes}

    for img_path in image_paths:
        img_pil = Image.open(img_path).convert("RGB")
        img_resized = img_pil.resize((imgsz, imgsz))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        label_path = data_root / "labels" / "base_train" / (img_path.stem + ".txt")
        labels = _load_labels(label_path)
        if not labels:
            continue

        scale_features = _extract_cv3_features(model, img_tensor)

        for cls_id, cx, cy, w, h in labels:
            cls_name = cls_idx_to_name.get(cls_id)
            if cls_name is None or cls_name not in base_set:
                continue

            x1, y1, x2, y2 = _cxcywh_to_xyxy(cx, cy, w, h, imgsz, imgsz)
            box_w, box_h = x2 - x1, y2 - y1
            if box_w < 2 or box_h < 2:
                continue

            box_area = box_w * box_h
            if box_area < 96 * 96:
                scale_idx = 0
            elif box_area < 192 * 192:
                scale_idx = 1
            else:
                scale_idx = 2

            feat_map = scale_features[scale_idx]
            stride = imgsz / feat_map.shape[-1]
            rois = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32, device=device)
            pooled = roi_align(feat_map, rois, output_size=(1, 1), spatial_scale=1.0 / stride)
            class_features[cls_name].append(pooled.squeeze())

    prototypes: dict[str, torch.Tensor] = {}
    for cls_name in base_classes:
        feats = class_features[cls_name]
        if not feats:
            c3 = next((f.shape[0] for v in class_features.values() for f in v), 128)
            print(f"  WARNING: No features for base class '{cls_name}'. Using zero vector.")
            prototypes[cls_name] = torch.zeros(c3)
        else:
            stacked = torch.stack(feats)
            prototypes[cls_name] = stacked.mean(dim=0).cpu()
            print(f"  {cls_name}: {len(feats)} instances → norm={prototypes[cls_name].norm():.3f}")

    del model
    torch.cuda.empty_cache()

    return prototypes


# ---------------------------------------------------------------------------
# 3. Adaptation MLP
# ---------------------------------------------------------------------------

class AdaptationMLP(nn.Module):
    """Lightweight MLP mapping Florence-2 text embeddings to YOLO visual feature space."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_adaptation_mlp(
    text_embeddings: dict[str, torch.Tensor],
    visual_prototypes: dict[str, torch.Tensor],
    base_classes: list[str],
    hidden_dim: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    epochs: int = 1000,
    device: str = "cuda:0",
) -> AdaptationMLP:
    """Train adaptation MLP on base class (text_embedding, visual_prototype) pairs.

    Minimizes cosine distance: loss = mean(1 - cos_sim(MLP(text_emb), proto)).
    Input/output dims are auto-detected from the actual embeddings.

    Args:
        text_embeddings: class_name → Florence-2 embedding.
        visual_prototypes: class_name → YOLO cv3 prototype.
        base_classes: Base class names to train on.
        hidden_dim: MLP hidden layer dim.
        lr: Learning rate.
        weight_decay: L2 regularization (important — only 15 training pairs).
        epochs: Training epochs.
        device: Torch device.

    Returns:
        Trained AdaptationMLP in eval mode.
    """
    # Build training tensors: (N_base, input_dim) and (N_base, output_dim)
    X = torch.stack([text_embeddings[c] for c in base_classes]).to(device)
    Y = torch.stack([visual_prototypes[c] for c in base_classes]).to(device)

    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    # L2-normalize targets (consistent with CosineConv2d weight normalization)
    Y = F.normalize(Y, dim=1)

    mlp = AdaptationMLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"  Training adaptation MLP ({input_dim}→{hidden_dim}→{output_dim}) "
          f"on {len(base_classes)} base classes for {epochs} epochs...")

    mlp.train()
    for epoch in range(epochs):
        pred = mlp(X)
        pred_norm = F.normalize(pred, dim=1)

        # Cosine similarity loss: maximize similarity
        cos_sim = (pred_norm * Y).sum(dim=1)
        loss = (1 - cos_sim).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}: loss={loss.item():.4f}, "
                  f"mean_cos_sim={cos_sim.mean().item():.4f}")

    mlp.eval()
    return mlp


# ---------------------------------------------------------------------------
# 4. CosineConv2d Weight Initialization
# ---------------------------------------------------------------------------

@torch.no_grad()
def init_cosine_head_with_florence2(
    model: YOLO,
    mlp: AdaptationMLP,
    text_embeddings: dict[str, torch.Tensor],
    novel_classes: list[str],
    all_classes: list[str],
) -> None:
    """Initialize CosineConv2d novel class weights using Florence-2 adapted embeddings.

    For each novel class:
      1. Look up its Florence-2 text embedding (1024-dim)
      2. Pass through trained adaptation MLP → c3-dim weight vector
      3. L2-normalize and assign to CosineConv2d weight row

    Base class weights are left unchanged (transferred from base pretrain).

    Args:
        model: YOLO model with FSODDetect head (weights already loaded).
        mlp: Trained AdaptationMLP.
        text_embeddings: class_name → Florence-2 embedding (1024-dim).
        novel_classes: Novel class names to initialize.
        all_classes: All class names (for index mapping).
    """
    cls_name_to_idx = {name: i for i, name in enumerate(all_classes)}
    detect = model.model.model[-1]
    device = next(mlp.parameters()).device

    # Generate novel class weight vectors via MLP
    novel_weights: dict[str, torch.Tensor] = {}
    for cls_name in novel_classes:
        emb = text_embeddings[cls_name].unsqueeze(0).to(device)  # (1, 1024)
        weight_vec = mlp(emb).squeeze(0)  # (c3,)
        weight_vec = F.normalize(weight_vec.unsqueeze(0), dim=1).squeeze(0)
        novel_weights[cls_name] = weight_vec

    for i in range(detect.nl):
        cosine_layer = detect.cv3[i][-1]  # CosineConv2d
        weight = cosine_layer.weight.data

        for cls_name in novel_classes:
            idx = cls_name_to_idx.get(cls_name)
            if idx is None:
                continue
            weight[idx] = novel_weights[cls_name].to(weight.device)

        print(f"  Scale {i}: initialized {len(novel_classes)} novel class weights "
              f"via Florence-2 adaptation")
