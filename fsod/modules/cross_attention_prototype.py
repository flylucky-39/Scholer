"""Cross-Attention Prototype (CAP) for multi-modal prototype enhancement.

Replaces simple feature averaging with bidirectional cross-attention between
spatial visual features and VLM text embeddings, producing prototypes that
preserve spatial structure and semantic context.

Idea: instead of RoIAlign 1x1 → average → alpha-blend with text,
we RoIAlign 3x3 to preserve spatial layout, then use cross-attention:
  text → Q attends to visual K,V  (semantic-guided visual attention)
  visual → Q attends to text K,V   (visual-contextualized text)
  → concat → project → L2-normalized fused prototype

Trained on BASE classes where target = base-pretrained Conv2d weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from pathlib import Path
from PIL import Image
import numpy as np

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 1. Cross-Attention Building Blocks
# ---------------------------------------------------------------------------

class CrossAttentionLayer(nn.Module):
    """Multi-head cross-attention: Q attends to K,V."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, f"dim {dim} not divisible by n_heads {n_heads}"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Args:
            q: (B, Lq, D)
            k: (B, Lk, D)
            v: (B, Lv, D)  (Lv == Lk)
        Returns:
            (B, Lq, D)
        """
        B, Lq, D = q.shape
        Lk = k.shape[1]

        Q = self.q_proj(q).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out_proj(out)


class CrossAttentionPrototype(nn.Module):
    """Bidirectional cross-attention fusion for prototype enhancement.

    Produces a L2-normalized fused prototype from spatial visual features
    and a VLM text embedding via bidirectional cross-attention.
    """

    def __init__(self, visual_dim: int = 256, text_dim: int = 1024,
                 hidden_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_to_vis = CrossAttentionLayer(hidden_dim, n_heads)
        self.vis_to_text = CrossAttentionLayer(hidden_dim, n_heads)
        self.out_proj = nn.Linear(hidden_dim * 2, visual_dim)

    def forward(self, visual_feat: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Args:
            visual_feat: (B, C_vis, H, W) spatial feature map (e.g. 256, 3, 3)
            text_emb:    (B, C_text) text embedding (e.g. 1024)
        Returns:
            (B, C_vis) L2-normalized fused prototype
        """
        B, C, H, W = visual_feat.shape
        vis_tokens = visual_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)

        vis_tokens = self.vis_proj(vis_tokens)               # (B, H*W, hidden)
        text_tokens = self.text_proj(text_emb).unsqueeze(1)  # (B, 1, hidden)

        # Text → Visual: text token queries visual tokens
        text_attended = self.text_to_vis(text_tokens, vis_tokens, vis_tokens)  # (B, 1, hidden)

        # Visual → Text: visual tokens query text token
        vis_attended = self.vis_to_text(vis_tokens, text_tokens, text_tokens)  # (B, H*W, hidden)
        vis_pooled = vis_attended.mean(dim=1)                                  # (B, hidden)

        # Fuse both directions
        fused = torch.cat([text_attended.squeeze(1), vis_pooled], dim=-1)      # (B, 2*hidden)
        proto = self.out_proj(fused)                                           # (B, C_vis)
        return F.normalize(proto, dim=1)


# ---------------------------------------------------------------------------
# 2. Spatial prototype extraction (RoIAlign 3x3 instead of 1x1)
# ---------------------------------------------------------------------------

def _load_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO-format labels."""
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
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


@torch.no_grad()
def _extract_cv3_features(model: YOLO, img_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Run forward pass and extract penultimate cv3 features per scale."""
    net = model.model
    detect = net.model[-1]

    y = []
    x = img_tensor
    for m in net.model:
        if m is detect:
            break
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in net.save else None)

    if isinstance(detect.f, list):
        head_inputs = [y[j] for j in detect.f]
    else:
        head_inputs = [x]

    features = []
    for i, xi in enumerate(head_inputs):
        feat = xi
        cv3_seq = detect.cv3[i]
        for layer in list(cv3_seq)[:-1]:
            feat = layer(feat)
        features.append(feat)
    return features


@torch.no_grad()
def extract_spatial_prototypes(
    base_weights: str | Path,
    data_root: str | Path,
    target_classes: list[str],
    all_classes: list[str],
    imgsz: int = 640,
    device: str = "cuda:0",
    roi_size: int = 3,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-class spatial feature maps from support set.

    Unlike the original prototype.py which pools to 1x1, this preserves
    spatial structure by RoIAlign at roi_size × roi_size.

    Returns:
        Dict class_name → list of spatial feature maps (C_vis, roi_size, roi_size).
        The list may have entries from multiple images / multiple scales.
    """
    data_root = Path(data_root)
    base_weights = Path(base_weights)

    model = YOLO(str(base_weights))
    model.to(device)
    model.model.eval()

    cls_idx_to_name = {i: name for i, name in enumerate(all_classes)}
    target_set = set(target_classes)

    manifest_path = data_root / "manifests" / "novel_finetune.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    image_paths = [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()]

    class_features: dict[str, list[torch.Tensor]] = {name: [] for name in target_classes}

    for img_path in image_paths:
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        img_resized = img_pil.resize((imgsz, imgsz))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        label_path = data_root / "labels" / "novel_finetune" / (img_path.stem + ".txt")
        labels = _load_labels(label_path)
        if not labels:
            continue

        scale_features = _extract_cv3_features(model, img_tensor)

        for cls_id, cx, cy, w, h in labels:
            cls_name = cls_idx_to_name.get(cls_id)
            if cls_name is None or cls_name not in target_set:
                continue

            x1, y1, x2, y2 = _cxcywh_to_xyxy(cx, cy, w, h, imgsz, imgsz)
            box_w, box_h = x2 - x1, y2 - y1
            if box_w < 2 or box_h < 2:
                continue

            # Pick scale based on box area
            box_area = box_w * box_h
            if box_area < 96 * 96:
                scale_idx = 0  # P3/8
            elif box_area < 192 * 192:
                scale_idx = 1  # P4/16
            else:
                scale_idx = 2  # P5/32

            feat_map = scale_features[scale_idx]
            stride = imgsz / feat_map.shape[-1]

            rois = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32, device=device)
            pooled = roi_align(feat_map, rois, output_size=(roi_size, roi_size),
                               spatial_scale=1.0 / stride)  # (1, C, roi, roi)
            class_features[cls_name].append(pooled.squeeze(0))  # (C, roi, roi)

    del model
    torch.cuda.empty_cache()
    return class_features


@torch.no_grad()
def extract_spatial_prototypes_base(
    base_weights: str | Path,
    data_root: str | Path,
    base_classes: list[str],
    all_classes: list[str],
    imgsz: int = 640,
    device: str = "cuda:0",
    roi_size: int = 3,
    max_images: int = 500,
    seed: int = 3407,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-class spatial feature maps from base_train split."""
    import random
    data_root = Path(data_root)
    manifest_path = data_root / "manifests" / "base_train.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Base train manifest not found: {manifest_path}")

    image_paths = [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()]
    if len(image_paths) > max_images:
        rng = random.Random(seed)
        image_paths = rng.sample(image_paths, max_images)

    print(f"  Extracting base spatial features from {len(image_paths)} images...")

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
            pooled = roi_align(feat_map, rois, output_size=(roi_size, roi_size),
                               spatial_scale=1.0 / stride)
            class_features[cls_name].append(pooled.squeeze(0))

    del model
    torch.cuda.empty_cache()
    return class_features


# ---------------------------------------------------------------------------
# 3. Training CAP on base classes
# ---------------------------------------------------------------------------

def _average_spatial_features(
    class_features: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average per-class spatial feature maps → one feature map per class."""
    result = {}
    for cls_name, feats in class_features.items():
        if not feats:
            continue
        result[cls_name] = torch.stack(feats).mean(dim=0)  # (C, roi, roi)
    return result


def train_cap_on_base(
    cap: CrossAttentionPrototype,
    text_embeddings: dict[str, torch.Tensor],
    spatial_features: dict[str, torch.Tensor],      # averaged per-class spatial features
    target_weights: dict[str, torch.Tensor],         # base-pretrained Conv2d weight per class per scale
    base_classes: list[str],
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    epochs: int = 300,
    device: str = "cuda:0",
) -> CrossAttentionPrototype:
    """Train CAP module on base classes.

    For each base class, CAP(visual_feat_3x3, text_emb) → predicted prototype.
    Target is the L2-normalized base-pretrained Conv2d weight for that class.
    Loss = 1 - cos_sim(pred, target).
    """
    cap = cap.to(device)
    cap.train()

    # Build training tensors
    X_text = torch.stack([text_embeddings[c] for c in base_classes]).to(device)       # (N, text_dim)
    X_vis = torch.stack([spatial_features[c] for c in base_classes]).to(device)       # (N, C, roi, roi)
    Y = torch.stack([target_weights[c] for c in base_classes]).to(device)             # (N, C)
    Y = F.normalize(Y, dim=1)

    optimizer = torch.optim.AdamW(cap.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"  Training Cross-Attention Prototype on {len(base_classes)} base classes "
          f"for {epochs} epochs (lr={lr}, wd={weight_decay})...")

    for epoch in range(epochs):
        pred = cap(X_vis, X_text)
        cos_sim = (pred * Y).sum(dim=1)
        loss = (1 - cos_sim).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}: loss={loss.item():.4f}, "
                  f"mean_cos_sim={cos_sim.mean().item():.4f}")

    cap.eval()
    return cap


@torch.no_grad()
def extract_target_weights_for_cap(
    base_weights: str | Path,
    target_classes: list[str],
    all_classes: list[str],
    scale_idx: int,
) -> dict[str, torch.Tensor]:
    """Extract per-class Conv2d weights from a specific scale of the base-pretrained model."""
    model = YOLO(str(base_weights))
    detect = model.model.model[-1]

    cls_name_to_idx = {name: i for i, name in enumerate(all_classes)}
    last_conv = detect.cv3[scale_idx][-1]
    w = last_conv.weight.data
    if w.dim() == 4:
        w = w.squeeze(-1).squeeze(-1)

    weights = {}
    for cls_name in target_classes:
        idx = cls_name_to_idx.get(cls_name)
        if idx is not None:
            weights[cls_name] = F.normalize(w[idx:idx + 1], dim=1).squeeze(0).cpu()

    del model
    torch.cuda.empty_cache()
    return weights


# ---------------------------------------------------------------------------
# 4. CosineConv2d initialization with CAP
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_text_embeddings_for_classes(
    florence2_path: str | Path,
    class_names: list[str],
    device: str = "cuda:0",
) -> dict[str, torch.Tensor]:
    """Extract Florence-2 text embeddings for a list of classes.

    Uses the approach from adaptation.py: mean-pool BART encoder outputs.
    """
    from fsod.florence2 import resolve_local_model_dir
    from transformers import AutoModelForCausalLM, AutoProcessor

    resolved_path = str(resolve_local_model_dir(florence2_path))
    processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path, dtype=torch.float32, trust_remote_code=True,
        local_files_only=True, attn_implementation="eager",
    ).to(device)
    model.eval()
    encoder = model.language_model.get_encoder()

    embeddings = {}
    with torch.no_grad():
        for name in class_names:
            tokens = processor.tokenizer(name, return_tensors="pt", padding=False).to(device)
            enc_out = encoder(input_ids=tokens["input_ids"], return_dict=True)
            emb = enc_out.last_hidden_state.mean(dim=1).squeeze(0)
            embeddings[name] = emb.cpu()

    del model, encoder, processor
    torch.cuda.empty_cache()
    return embeddings


@torch.no_grad()
def init_cosine_head_with_cap(
    model: YOLO,
    cap_models: list[CrossAttentionPrototype],
    text_embeddings: dict[str, torch.Tensor],
    spatial_prototypes: dict[str, list[torch.Tensor]],   # per-class list of (C, roi, roi)
    novel_classes: list[str],
    all_classes: list[str],
    alpha: float = 0.5,
    fusion_mode: str = "learnable",
) -> None:
    """Initialize CosineConv2d weights using CAP fused prototypes.

    For each scale:
      1. Average spatial features per novel class
      2. CAP(avg_spatial_feat, text_emb) → fused prototype
      3. Initialize weight row with fused prototype

    In 'learnable' or 'fixed' mode, a visual prototype prior is also registered
    via set_weight_prior for training-time fusion.
    """
    cls_name_to_idx = {name: i for i, name in enumerate(all_classes)}
    detect = model.model.model[-1]
    device = next(cap_models[0].parameters()).device

    for i in range(detect.nl):
        cap = cap_models[i]
        cosine_layer = detect.cv3[i][-1]
        weight = cosine_layer.weight.data
        prior_indices: list[int] = []
        prior_weights: list[torch.Tensor] = []

        for cls_name in novel_classes:
            idx = cls_name_to_idx.get(cls_name)
            if idx is None:
                continue

            # Get text embedding
            text_emb = text_embeddings[cls_name].unsqueeze(0).to(device)  # (1, text_dim)

            # Get averaged spatial visual feature
            feats = spatial_prototypes.get(cls_name, [])
            if not feats:
                print(f"  WARNING: No spatial features for '{cls_name}', skipping CAP init")
                continue
            avg_feat = torch.stack(feats).mean(dim=0).unsqueeze(0).to(device)  # (1, C, roi, roi)

            # CAP forward → fused prototype
            fused_proto = cap(avg_feat, text_emb)  # (1, C_vis)

            if fusion_mode == "init_only":
                weight[idx] = fused_proto.squeeze(0).to(weight.device)
                continue

            weight[idx] = fused_proto.squeeze(0).to(weight.device)
            prior_indices.append(idx)

            # Visual prototype prior (simple average, no cross-attention)
            proto_prior = avg_feat.mean(dim=[2, 3])  # (1, C) spatial pooling
            proto_prior = F.normalize(proto_prior, dim=1).squeeze(0)
            prior_weights.append(proto_prior.to(weight.device))

        if prior_indices:
            cosine_layer.set_weight_prior(
                prior_indices,
                torch.stack(prior_weights),
                alpha_init=alpha,
                fusion_mode=fusion_mode,
            )

        mode_desc = (
            "init-only"
            if fusion_mode == "init_only"
            else f"{fusion_mode} alpha={cosine_layer.get_blend_alpha():.2f}"
        )
        print(f"  Scale {i}: initialized {len(prior_indices)} novel classes via CAP ({mode_desc})")
