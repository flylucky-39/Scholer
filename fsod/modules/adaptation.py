"""Adaptation layer: Florence-2 → CosineConv2d weight initialization.

Two approaches implemented:

A) Simple MLP (original Exp 3):
   Florence-2 text encoder encodes class names → MLP maps to visual space
   → CosineConv2d weight init.

B) Textual Inversion + FiLM modulation (revised Exp 3):
   Florence-2 captions cropped objects → text encoder → description embeddings.
   FiLM network modulates visual prototypes with description embeddings.
   Trained on base classes where target = base-pretrained Conv2d weights.
   Applied to novel classes → CosineConv2d weight init.

At inference time: zero extra cost — Florence-2 is only used during initialization.
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


@torch.no_grad()
def init_cosine_head_fused(
    model: YOLO,
    mlp: AdaptationMLP,
    text_embeddings: dict[str, torch.Tensor],
    visual_prototypes: dict[str, torch.Tensor],
    novel_classes: list[str],
    all_classes: list[str],
    alpha: float = 0.5,
) -> None:
    """Initialize CosineConv2d with fused Florence-2 + visual prototype weights.

    For each novel class:
      w_final = normalize( alpha * w_florence + (1 - alpha) * w_proto )

    Args:
        model: YOLO model with FSODDetect head.
        mlp: Trained AdaptationMLP.
        text_embeddings: class_name → Florence-2 text embedding.
        visual_prototypes: class_name → visual prototype from support set (c3-dim).
        novel_classes: Novel class names.
        all_classes: All class names.
        alpha: Blend weight (1.0 = pure Florence-2, 0.0 = pure prototype).
    """
    cls_name_to_idx = {name: i for i, name in enumerate(all_classes)}
    detect = model.model.model[-1]
    device = next(mlp.parameters()).device

    # Generate Florence-2 adapted weights
    florence_weights: dict[str, torch.Tensor] = {}
    for cls_name in novel_classes:
        emb = text_embeddings[cls_name].unsqueeze(0).to(device)
        w = mlp(emb).squeeze(0)
        florence_weights[cls_name] = F.normalize(w.unsqueeze(0), dim=1).squeeze(0)

    for i in range(detect.nl):
        cosine_layer = detect.cv3[i][-1]
        weight = cosine_layer.weight.data

        for cls_name in novel_classes:
            idx = cls_name_to_idx.get(cls_name)
            if idx is None:
                continue

            w_f = florence_weights[cls_name].to(weight.device)
            proto = visual_prototypes.get(cls_name)
            if proto is not None:
                w_p = F.normalize(proto.unsqueeze(0).to(weight.device), dim=1).squeeze(0)
                w_fused = alpha * w_f + (1 - alpha) * w_p
                weight[idx] = F.normalize(w_fused.unsqueeze(0), dim=1).squeeze(0)
            else:
                weight[idx] = w_f

        print(f"  Scale {i}: initialized {len(novel_classes)} novel class weights "
              f"(fused α={alpha})")


# ===========================================================================
# 5. Textual Inversion: Description-based Embeddings + FiLM Modulation
# ===========================================================================

@torch.no_grad()
def generate_and_encode_descriptions(
    model_path: str | Path,
    data_root: str | Path,
    split_name: str,
    target_classes: list[str],
    all_classes: list[str],
    device: str = "cuda:0",
    max_crops_per_class: int = 30,
    min_crop_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Generate Florence-2 captions for cropped objects, encode as text embeddings.

    Textual Inversion inspired: instead of encoding bare class names, generate
    rich visual descriptions from actual images. This captures fine-grained
    visual attributes that a simple class name cannot convey.

    Pipeline per class:
      1. Crop GT bbox regions from training images
      2. Caption each crop with Florence-2 <DETAILED_CAPTION>
      3. Encode caption through Florence-2 BART text encoder (mean-pool)
      4. Average all caption embeddings → class-level description embedding

    Returns:
        Dict class_name → description embedding (e.g., 768-dim for Florence-2-base).
    """
    from fsod.florence2 import resolve_local_model_dir
    from fsod.modules.prototype import _load_labels
    from transformers import AutoModelForCausalLM, AutoProcessor

    resolved_path = str(resolve_local_model_dir(model_path))
    processor = AutoProcessor.from_pretrained(
        resolved_path, trust_remote_code=True, local_files_only=True,
    )
    fl_model = AutoModelForCausalLM.from_pretrained(
        resolved_path, dtype=torch.float32,
        trust_remote_code=True, local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    fl_model.eval()

    encoder = fl_model.language_model.get_encoder()

    data_root = Path(data_root)
    manifest_path = data_root / "manifests" / f"{split_name}.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    image_paths = [
        Path(line.strip())
        for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()
    ]

    cls_idx_to_name = {i: name for i, name in enumerate(all_classes)}
    target_set = set(target_classes)

    # --- Phase 1: Collect crops per class ---
    class_crops: dict[str, list[Image.Image]] = {name: [] for name in target_classes}

    for img_path in image_paths:
        label_path = data_root / "labels" / split_name / (img_path.stem + ".txt")
        labels = _load_labels(label_path)
        if not labels:
            continue

        has_target = any(
            cls_idx_to_name.get(cls_id) in target_set for cls_id, *_ in labels
        )
        if not has_target:
            continue

        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        for cls_id, cx, cy, w, h in labels:
            cls_name = cls_idx_to_name.get(cls_id)
            if cls_name is None or cls_name not in target_set:
                continue
            if len(class_crops[cls_name]) >= max_crops_per_class:
                continue

            x1 = max(0, int((cx - w / 2) * orig_w))
            y1 = max(0, int((cy - h / 2) * orig_h))
            x2 = min(orig_w, int((cx + w / 2) * orig_w))
            y2 = min(orig_h, int((cy + h / 2) * orig_h))

            if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                continue

            class_crops[cls_name].append(img_pil.crop((x1, y1, x2, y2)))

    # --- Phase 2: Caption each crop and encode ---
    task_prompt = "<DETAILED_CAPTION>"
    class_embeddings: dict[str, list[torch.Tensor]] = {name: [] for name in target_classes}
    total_crops = sum(len(v) for v in class_crops.values())
    processed = 0

    for cls_name in target_classes:
        crops = class_crops[cls_name]

        if not crops:
            # Fallback: encode bare class name
            print(f"  WARNING: No crops for '{cls_name}', using class name embedding")
            tokens = processor.tokenizer(
                cls_name, return_tensors="pt", padding=False,
            ).to(device)
            enc_out = encoder(input_ids=tokens["input_ids"], return_dict=True)
            class_embeddings[cls_name].append(
                enc_out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            )
            continue

        for crop in crops:
            # Generate caption
            inputs = processor(
                text=task_prompt, images=crop, return_tensors="pt",
            ).to(device)
            generated_ids = fl_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
                use_cache=False,
            )
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=False,
            )[0]
            parsed = processor.post_process_generation(
                generated_text, task=task_prompt, image_size=crop.size,
            )
            caption = parsed.get(task_prompt, cls_name)

            # Encode caption through text encoder
            tokens = processor.tokenizer(
                caption, return_tensors="pt", padding=False,
            ).to(device)
            enc_out = encoder(input_ids=tokens["input_ids"], return_dict=True)
            emb = enc_out.last_hidden_state.mean(dim=1).squeeze(0)
            class_embeddings[cls_name].append(emb.cpu())
            processed += 1

        print(f"  {cls_name}: {len(crops)} crops → {processed}/{total_crops} done")

    # --- Phase 3: Average per class ---
    result: dict[str, torch.Tensor] = {}
    emb_dim = None
    for cls_name in target_classes:
        embs = class_embeddings[cls_name]
        if embs:
            result[cls_name] = torch.stack(embs).mean(dim=0)
            emb_dim = result[cls_name].shape[0]
    for cls_name in target_classes:
        if cls_name not in result:
            result[cls_name] = torch.zeros(emb_dim or 768)

    del fl_model, encoder, processor
    torch.cuda.empty_cache()

    print(f"  Description embeddings: {len(result)} classes, dim={emb_dim}")
    return result


# ---------------------------------------------------------------------------
# 6. FiLM Modulation Network
# ---------------------------------------------------------------------------

class TextModulatedPrototype(nn.Module):
    """FiLM-style modulation: text description conditions scale & shift of visual prototype.

    output = proto * (1 + gamma(text)) + beta(text)
    """

    def __init__(self, text_dim: int = 768, proto_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proto_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proto_dim),
        )

    def forward(self, proto: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(text_emb)
        beta = self.beta_net(text_emb)
        return proto * (1 + gamma) + beta


@torch.no_grad()
def extract_target_weights(
    base_weights: str | Path,
    target_classes: list[str],
    all_classes: list[str],
) -> list[dict[str, torch.Tensor]]:
    """Extract per-scale Conv2d classification weights from base-pretrained YOLO model.

    These are the optimal classification directions learned during full base
    pretraining, used as supervision targets for the FiLM modulation network.
    """
    model = YOLO(str(base_weights))
    detect = model.model.model[-1]  # Standard Detect head

    cls_name_to_idx = {name: i for i, name in enumerate(all_classes)}

    scale_weights: list[dict[str, torch.Tensor]] = []
    for scale_idx in range(detect.nl):
        last_conv = detect.cv3[scale_idx][-1]
        w = last_conv.weight.data  # (nc, c3, 1, 1)
        if w.dim() == 4:
            w = w.squeeze(-1).squeeze(-1)  # → (nc, c3)

        weights = {}
        for cls_name in target_classes:
            idx = cls_name_to_idx.get(cls_name)
            if idx is not None:
                weights[cls_name] = F.normalize(w[idx:idx + 1], dim=1).squeeze(0).cpu()

        scale_weights.append(weights)
        print(f"  Scale {scale_idx}: extracted {len(weights)} target weights (dim={w.shape[1]})")

    del model
    torch.cuda.empty_cache()
    return scale_weights


def train_modulation_network(
    desc_embeddings: dict[str, torch.Tensor],
    visual_prototypes: dict[str, torch.Tensor],
    target_weights: list[dict[str, torch.Tensor]],
    base_classes: list[str],
    hidden_dim: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    epochs: int = 1000,
    device: str = "cuda:0",
) -> nn.ModuleList:
    """Train one FiLM modulation network per detection scale.

    Learns, for each scale s:
      FiLM_s(base_proto, base_description_emb) ≈ base_pretrained_weight_s

    Then for novel classes:
      FiLM_s(novel_proto, novel_desc_emb) → scale-specific novel weight
    """
    X_text = torch.stack([desc_embeddings[c] for c in base_classes]).to(device)
    X_proto = torch.stack([visual_prototypes[c] for c in base_classes]).to(device)
    X_proto = F.normalize(X_proto, dim=1)
    targets_by_scale = [torch.stack([scale_weights[c] for c in base_classes]).to(device) for scale_weights in target_weights]

    text_dim = X_text.shape[1]
    proto_dim = X_proto.shape[1]

    films = nn.ModuleList(
        [TextModulatedPrototype(text_dim, proto_dim, hidden_dim) for _ in targets_by_scale]
    ).to(device)
    optimizer = torch.optim.AdamW(films.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        f"  Training {len(films)} scale-specific FiLM heads "
        f"({text_dim}+{proto_dim}→{proto_dim}) on {len(base_classes)} base classes for {epochs} epochs..."
    )

    films.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        scale_losses = []
        scale_cos_sims = []
        for film, scale_targets in zip(films, targets_by_scale):
            pred = film(X_proto, X_text)
            pred_norm = F.normalize(pred, dim=1)

            cos_sim = (pred_norm * scale_targets).sum(dim=1)
            scale_losses.append((1 - cos_sim).mean())
            scale_cos_sims.append(cos_sim.mean().detach())

        loss = torch.stack(scale_losses).mean()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            mean_cos = torch.stack(scale_cos_sims)
            cos_str = ", ".join(f"P{scale_idx + 3}={value.item():.4f}" for scale_idx, value in enumerate(mean_cos))
            print(f"  Epoch {epoch + 1:4d}/{epochs}: loss={loss.item():.4f}, {cos_str}")

    films.eval()
    return films


# ---------------------------------------------------------------------------
# 7. CosineConv2d Init via Text-Modulated Prototypes
# ---------------------------------------------------------------------------

@torch.no_grad()
def init_cosine_head_modulated(
    model: YOLO,
    film: TextModulatedPrototype | nn.ModuleList | list[TextModulatedPrototype],
    desc_embeddings: dict[str, torch.Tensor],
    visual_prototypes: dict[str, torch.Tensor],
    novel_classes: list[str],
    all_classes: list[str],
    alpha: float = 0.5,
) -> None:
    """Initialize CosineConv2d novel class weights via scale-specific FiLM + learnable prototype prior.

    For each novel class:
      film_w  = normalize( FiLM_scale(novel_proto, novel_desc_emb) )
      raw_w   = normalize( novel_proto )

    The trainable branch is initialized with film_w, while raw_w is stored as a
    fixed prior inside each CosineConv2d layer. During finetuning the effective
    novel-class weight becomes:
      final_w = normalize( alpha_scale * raw_w + (1 - alpha_scale) * trainable_w )

    This adds exactly one learnable alpha per detection scale with zero extra
    data-loading or VLM inference cost during training.
    """
    cls_name_to_idx = {name: i for i, name in enumerate(all_classes)}
    detect = model.model.model[-1]
    if isinstance(film, TextModulatedPrototype):
        films = [film] * detect.nl
    else:
        films = list(film)
    if len(films) != detect.nl:
        raise ValueError(f"Expected {detect.nl} FiLM modules, got {len(films)}")

    device = next(films[0].parameters()).device

    for i in range(detect.nl):
        scale_film = films[i]
        cosine_layer = detect.cv3[i][-1]
        weight = cosine_layer.weight.data
        prior_indices: list[int] = []
        prior_weights: list[torch.Tensor] = []

        for cls_name in novel_classes:
            idx = cls_name_to_idx.get(cls_name)
            if idx is None:
                continue

            text_emb = desc_embeddings[cls_name].unsqueeze(0).to(device)
            proto = visual_prototypes[cls_name].unsqueeze(0).to(device)
            proto_norm = F.normalize(proto, dim=1)

            film_w = scale_film(proto_norm, text_emb).squeeze(0)
            film_w = F.normalize(film_w.unsqueeze(0), dim=1).squeeze(0)

            weight[idx] = film_w.to(weight.device)
            prior_indices.append(idx)
            prior_weights.append(proto_norm.squeeze(0).to(weight.device))

        if prior_indices:
            cosine_layer.set_weight_prior(prior_indices, torch.stack(prior_weights), alpha_init=alpha)

        print(f"  Scale {i}: initialized {len(novel_classes)} novel class weights "
              f"via text-modulated prototypes (learnable alpha init={cosine_layer.get_blend_alpha():.2f})")
