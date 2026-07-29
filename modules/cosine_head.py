"""Cosine Classifier head for YOLO few-shot detection.

Replaces the final 1x1 Conv2d classification layer in each scale of the
Detect head with a cosine-similarity-based classifier.  This makes the
classification decision based on the angle between the feature vector and
each class weight (prototype), which is more robust in low-data regimes.

Background Suppression (VLM-Guided):
    Anchor-free detectors predict classification scores at every spatial
    location.  When a background prototype b is available, CosineConv2d
    suppresses background-like features at inference time:

        score_c = exp(scale_c) * [cos(x, w_c) - gamma * cos(x, b)] + bias_c

    This penalises attention to background-correlated features with zero
    extra parameters in the forward path (gamma is a fixed scalar).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv, DWConv


class CosineConv2d(nn.Module):
    """1x1 convolution that uses cosine similarity instead of dot product.

    For each spatial location, computes:
        score_c = exp(scale_c) * cos(feature, weight_c) + bias_c

    where scale_c (temperature) and bias_c are learnable per-class parameters.
    The bias is initialised to the standard YOLO logit prior so that the
    initial BCE loss matches standard YOLO detection.

    When a background prototype is set (via ``set_background_proto``), the
    forward pass additionally subtracts the cosine similarity to the background
    prototype from all class scores:

        score_c = exp(scale_c) * [cos(x, w_c) - gamma * cos(x, b)] + bias_c

    This suppresses background-correlated features at inference time.

    NOTE: Because the bias dominates (≈−10) and cosine values are small
    (≈0.05), sigmoid(score) is compressed near 0.  This does NOT affect
    mAP (rank-invariant), but raw sigmoid confidences are not directly
    interpretable.  For visualisation / paper figures, use the
    ``cosine_score`` helper that returns the un-biased cosine score
    rescaled to [0, 1] via per-class min-max normalisation.
    """

    def __init__(self, in_channels: int, out_channels: int, temperature: float = 5.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        # Per-class learnable temperature (log-space for positivity)
        self.scale = nn.Parameter(torch.full((out_channels,), math.log(temperature)))
        self.blend_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("weight_prior", torch.zeros(out_channels, in_channels))
        self.register_buffer("weight_prior_mask", torch.zeros(out_channels, dtype=torch.bool))
        self.register_buffer("fixed_blend_alpha", torch.tensor(0.5))
        self.prior_fusion_mode = "learnable"
        # Background suppression
        self.register_buffer("bg_proto", torch.zeros(in_channels))
        self.register_buffer("bg_gamma", torch.tensor(0.0))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    # ── weight_prior helpers (unchanged) ──────────────────────────

    def set_weight_prior(
        self,
        class_indices: list[int],
        prior_weights: torch.Tensor,
        alpha_init: float = 0.5,
        fusion_mode: str = "learnable",
    ) -> None:
        self.weight_prior.zero_()
        self.weight_prior_mask.zero_()
        self.prior_fusion_mode = fusion_mode
        if not class_indices:
            return
        if fusion_mode not in {"learnable", "fixed", "init_only"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if prior_weights.ndim != 2 or prior_weights.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected prior_weights shape (N, {self.in_channels}), got {tuple(prior_weights.shape)}"
            )
        prior_weights = prior_weights.to(device=self.weight_prior.device, dtype=self.weight_prior.dtype)
        self.weight_prior[class_indices] = prior_weights
        self.weight_prior_mask[class_indices] = True
        alpha_init = min(max(alpha_init, 1e-4), 1 - 1e-4)
        self.blend_logit.data.fill_(math.log(alpha_init / (1 - alpha_init)))
        self.fixed_blend_alpha.fill_(alpha_init)
        self.blend_logit.requires_grad_(fusion_mode == "learnable")
        if fusion_mode == "init_only":
            self.weight_prior_mask.zero_()

    def has_active_prior(self) -> bool:
        return bool(self.weight_prior_mask.any().item())

    def get_blend_alpha(self) -> float:
        if self.prior_fusion_mode == "fixed":
            return float(self.fixed_blend_alpha.item())
        return torch.sigmoid(self.blend_logit.detach()).item()

    # ── background suppression helpers ────────────────────────────

    def set_background_proto(self, bg_proto: torch.Tensor, gamma: float = 0.3) -> None:
        """Set the background prototype for inference-time background suppression.

        Args:
            bg_proto: Background prototype tensor of shape (in_channels,).
            gamma: Suppression strength — how much to subtract the background
                   cosine similarity from all class scores.  0 = disabled.
        """
        if bg_proto.shape != (self.in_channels,):
            raise ValueError(
                f"Expected bg_proto shape ({self.in_channels},), got {tuple(bg_proto.shape)}"
            )
        self.bg_proto.copy_(bg_proto.to(device=self.bg_proto.device, dtype=self.bg_proto.dtype))
        self.bg_gamma.fill_(gamma)

    @property
    def has_bg_suppression(self) -> bool:
        return bool(self.bg_gamma.item() > 0 and self.bg_proto.norm().item() > 0)

    # ── forward ───────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return score = exp(scale) * [cos(x, w) - gamma * cos(x, bg)] + bias.

        When background suppression is active (bg_gamma > 0), the cosine
        similarity to the background prototype is subtracted from every
        class score, penalising background-like features.
        """
        w_norm = F.normalize(self.weight, dim=1)
        if self.weight_prior_mask.any():
            prior_norm = F.normalize(self.weight_prior, dim=1)
            alpha = (
                self.fixed_blend_alpha
                if self.prior_fusion_mode == "fixed"
                else torch.sigmoid(self.blend_logit)
            )
            fused = alpha * prior_norm + (1 - alpha) * w_norm
            fused = F.normalize(fused, dim=1)
            mask = self.weight_prior_mask.view(-1, 1)
            w_norm = torch.where(mask, fused, w_norm)
        x_norm = F.normalize(x, dim=1)
        cos_sim = F.conv2d(x_norm, w_norm.unsqueeze(-1).unsqueeze(-1))

        # Background suppression: subtract gamma * cos(x, bg_proto) from all classes
        if self.has_bg_suppression:
            bg_norm = F.normalize(self.bg_proto, dim=0)
            bg_cos = F.conv2d(x_norm, bg_norm.view(1, self.in_channels, 1, 1))
            # bg_cos shape: (B, 1, H, W) — broadcast to (B, nc, H, W)
            cos_sim = cos_sim - self.bg_gamma * bg_cos

        return cos_sim * self.scale.exp().view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class FSODDetect(Detect):
    """YOLO Detect head with Cosine Classifier replacing the final classification layer.

    Inherits from Detect and only modifies cv3 (classification branch):
    - The last nn.Conv2d(c3, nc, 1) is replaced with CosineConv2d(c3, nc).
    - Everything else (bbox regression, DFL, anchors, strides) is unchanged.
    """

    def __init__(self, nc: int = 80, ch: tuple = (), temperature: float = 5.0) -> None:
        super().__init__(nc=nc, ch=ch)
        for i in range(self.nl):
            old_seq = self.cv3[i]
            c3_in = self._get_last_conv_in_channels(old_seq)
            old_seq[-1] = CosineConv2d(c3_in, self.nc, temperature=temperature)

    @staticmethod
    def _get_last_conv_in_channels(seq: nn.Sequential) -> int:
        last = seq[-1]
        if isinstance(last, nn.Conv2d):
            return last.in_channels
        raise TypeError(f"Expected last layer to be Conv2d, got {type(last)}")

    def set_background_proto(self, bg_proto: torch.Tensor, gamma: float = 0.3) -> None:
        """Inject background prototype into all CosineConv2d layers.

        Args:
            bg_proto: Background prototype tensor of shape (c3,).
            gamma: Background suppression strength.
        """
        for i in range(self.nl):
            cosine_layer = self.cv3[i][-1]
            if isinstance(cosine_layer, CosineConv2d):
                cosine_layer.set_background_proto(bg_proto, gamma)
        print(f"  Injected background prototype into {self.nl} scales (gamma={gamma})")

    def bias_init(self) -> None:
        """Standard YOLO bias initialisation (same as parent)."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)  # cls
