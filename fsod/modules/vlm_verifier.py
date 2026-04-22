"""CLIP-based per-box verifier for Direction C dual-path inference.

Given an image and a set of YOLO candidate boxes, crop each box (with optional
context padding), encode crops with the CLIP image encoder, and compute cosine
similarity against the pre-computed per-class text prototypes.

Designed to be model-agnostic on the YOLO side: it only consumes ``(image,
boxes_xyxy)`` and returns ``[N_boxes, num_classes]`` raw cosine similarities
in [-1, 1]. Score normalization and fusion live in ``dual_path_fusion``.

Typical usage (inference loop):
    verifier = CLIPVerifier(
        text_proto_path="runs/vlm_assets/dior_clip_vitb32_text_proto.pt",
        device="cuda",
    )
    sims = verifier.verify(image_chw_uint8, boxes_xyxy)   # [N, C]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# CLIP's standard ImageNet-style normalization (matches openai/clip-* and HF).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPVerifier:
    """Per-box CLIP verifier.

    Parameters
    ----------
    text_proto_path : str | Path
        Path to ``.pt`` produced by ``scripts/extract_vlm_text_proto.py``.
    model_name : str, optional
        Override the CLIP model id (defaults to the one stored in the proto file).
    device : str
        ``"cuda"`` or ``"cpu"``.
    pad_ratio : float
        Box context padding multiplier (1.0 = tight, 1.2 = +20% on each side).
    input_size : int
        CLIP input resolution (224 for ViT-B/32, 256 for some larger variants).
    batch_size : int
        Crop encode batch size (smaller = less peak VRAM).
    """

    def __init__(
        self,
        text_proto_path: str | Path,
        model_name: Optional[str] = None,
        device: str = "cuda",
        pad_ratio: float = 1.2,
        input_size: int = 224,
        batch_size: int = 64,
    ) -> None:
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
        self.device = torch.device(device)
        self.pad_ratio = float(pad_ratio)
        self.input_size = int(input_size)
        self.batch_size = int(batch_size)

        proto_path = Path(text_proto_path).expanduser()
        if not proto_path.exists():
            raise FileNotFoundError(f"Text prototype file not found: {proto_path}")
        payload = torch.load(proto_path, map_location="cpu")

        self.text_proto: Tensor = payload["proto"].to(self.device)              # [C, D]
        self.class_names: list[str] = list(payload["class_names"])
        self.class_name_to_idx: dict[str, int] = dict(payload["class_name_to_idx"])
        resolved_model_name: str = model_name or payload["model_name"]

        from transformers import CLIPModel
        self._clip = CLIPModel.from_pretrained(resolved_model_name).to(self.device).eval()
        # Freeze for inference.
        for p in self._clip.parameters():
            p.requires_grad_(False)

        self._mean = torch.tensor(_CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_CLIP_STD, device=self.device).view(1, 3, 1, 1)

        # Sanity: text proto dim must match the CLIP image encoder output dim.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
            dummy = (dummy - self._mean) / self._std
            test_feat = self._clip.get_image_features(pixel_values=dummy)
        if test_feat.shape[-1] != self.text_proto.shape[-1]:
            raise ValueError(
                f"CLIP image embed dim {test_feat.shape[-1]} != text proto dim "
                f"{self.text_proto.shape[-1]}. Re-run extract_vlm_text_proto.py "
                f"with the same --model as this verifier."
            )

    # ------------------------------------------------------------------
    # Class-index alignment helper
    # ------------------------------------------------------------------
    def assert_class_alignment(self, yolo_names: dict[int, str] | list[str]) -> None:
        """Raise if the verifier's class order does not match the YOLO model's.

        ``yolo_names`` may be Ultralytics' ``model.names`` (dict) or a plain list.
        """
        if isinstance(yolo_names, dict):
            ordered = [yolo_names[i] for i in sorted(yolo_names)]
        else:
            ordered = list(yolo_names)
        if ordered != self.class_names:
            raise ValueError(
                "Class index mismatch between YOLO and VLM verifier:\n"
                f"  YOLO:     {ordered}\n"
                f"  Verifier: {self.class_names}"
            )

    # ------------------------------------------------------------------
    # Crop + encode + score
    # ------------------------------------------------------------------
    @torch.no_grad()
    def verify(
        self,
        image: Tensor,
        boxes_xyxy: Tensor,
    ) -> Tensor:
        """Score every box against every class.

        Parameters
        ----------
        image : Tensor
            Either ``[3, H, W]`` float in [0, 1] (already on any device) or
            ``[3, H, W]`` uint8 in [0, 255].
        boxes_xyxy : Tensor
            ``[N, 4]`` in pixel coordinates of ``image``.

        Returns
        -------
        Tensor
            ``[N, C]`` raw cosine similarity in [-1, 1]. Empty tensor if N == 0.
        """
        if boxes_xyxy.numel() == 0:
            return torch.empty(0, self.text_proto.shape[0], device=self.device)

        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        image = image.to(self.device, non_blocking=True)
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"image must be [3, H, W], got {tuple(image.shape)}")

        H, W = image.shape[-2:]
        boxes = boxes_xyxy.to(self.device).float()

        crops = self._crop_with_padding(image, boxes, H, W)   # [N, 3, S, S]
        crops = (crops - self._mean) / self._std

        # Encode in mini-batches.
        feats: list[Tensor] = []
        for i in range(0, crops.shape[0], self.batch_size):
            chunk = crops[i : i + self.batch_size]
            feat = self._clip.get_image_features(pixel_values=chunk)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            feats.append(feat)
        feats_t = torch.cat(feats, dim=0)                       # [N, D]

        sims = feats_t @ self.text_proto.T                      # [N, C]
        return sims

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _crop_with_padding(
        self,
        image: Tensor,
        boxes: Tensor,
        H: int,
        W: int,
    ) -> Tensor:
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        bw = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0) * self.pad_ratio
        bh = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0) * self.pad_ratio

        x1 = (cx - bw * 0.5).clamp(min=0).long()
        y1 = (cy - bh * 0.5).clamp(min=0).long()
        x2 = (cx + bw * 0.5).clamp(max=W).long()
        y2 = (cy + bh * 0.5).clamp(max=H).long()

        crops_resized: list[Tensor] = []
        for i in range(boxes.shape[0]):
            xi1, yi1, xi2, yi2 = int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])
            if xi2 <= xi1 or yi2 <= yi1:
                # Degenerate box; emit a zeroed crop so downstream sim ≈ 0.
                crops_resized.append(
                    torch.zeros(3, self.input_size, self.input_size, device=image.device)
                )
                continue
            crop = image[:, yi1:yi2, xi1:xi2].unsqueeze(0)
            crop = F.interpolate(
                crop,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            crops_resized.append(crop)
        return torch.stack(crops_resized, dim=0)
