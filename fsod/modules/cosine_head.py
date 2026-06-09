"""Cosine Classifier head for YOLO few-shot detection.

Replaces the final 1x1 Conv2d classification layer in each scale of the
Detect head with a cosine-similarity-based classifier.  This makes the
classification decision based on the angle between the feature vector and
each class weight (prototype), which is more robust in low-data regimes.
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
        score_c = exp(temperature_c) * cos(feature, weight_c) + bias_c

    where bias_c and temperature_c are learnable per-class parameters.
    The bias is initialised to the standard YOLO logit prior (negative)
    so that the initial BCE loss stays in the same range as standard
    Conv2d classifiers.
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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def set_weight_prior(
        self,
        class_indices: list[int],
        prior_weights: torch.Tensor,
        alpha_init: float = 0.5,
        fusion_mode: str = "learnable",
    ) -> None:
        """Register fixed prototype priors for selected class rows.

        During finetuning the effective classifier weight for masked rows becomes:
            normalize(alpha * prior + (1 - alpha) * trainable_weight)
        where alpha is either a single learnable scalar, a fixed scalar, or is
        disabled after initialization depending on fusion_mode.
        """
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
        """Whether prior fusion is active during the forward pass."""
        return bool(self.weight_prior_mask.any().item())

    def get_blend_alpha(self) -> float:
        """Return the current prior fusion weight for logging."""
        if self.prior_fusion_mode == "fixed":
            return float(self.fixed_blend_alpha.item())
        return torch.sigmoid(self.blend_logit.detach()).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            (B, C_out, H, W)  cosine similarity scores scaled by temperature + bias
        """
        # Normalize weight: (C_out, C_in)
        w_norm = F.normalize(self.weight, dim=1)
        if self.weight_prior_mask.any():
            prior_norm = F.normalize(self.weight_prior, dim=1)
            if self.prior_fusion_mode == "fixed":
                alpha = self.fixed_blend_alpha
            else:
                alpha = torch.sigmoid(self.blend_logit)
            fused = alpha * prior_norm + (1 - alpha) * w_norm
            fused = F.normalize(fused, dim=1)
            mask = self.weight_prior_mask.view(-1, 1)
            w_norm = torch.where(mask, fused, w_norm)
        # Normalize input along channel dim: (B, C_in, H, W)
        x_norm = F.normalize(x, dim=1)
        # 1x1 conv with normalized weight → cosine similarity
        cos_sim = F.conv2d(x_norm, w_norm.unsqueeze(-1).unsqueeze(-1))
        return cos_sim * self.scale.exp().view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class FSODDetect(Detect):
    """YOLO Detect head with Cosine Classifier replacing the final classification layer.

    Inherits from Detect and only modifies cv3 (classification branch):
    - The last nn.Conv2d(c3, nc, 1) is replaced with CosineConv2d(c3, nc).
    - Everything else (bbox regression, DFL, anchors, strides) is unchanged.

    During inference, class scores are debiased before sigmoid so that
    confidence reflects the raw cosine similarity, not the logit-space
    score that includes a large negative training bias.
    """

    def __init__(self, nc: int = 80, ch: tuple = (), temperature: float = 5.0) -> None:
        super().__init__(nc=nc, ch=ch)
        self.temperature = temperature
        # Replace the last layer of each cv3 branch with CosineConv2d
        for i in range(self.nl):
            old_seq = self.cv3[i]
            c3_in = self._get_last_conv_in_channels(old_seq)
            old_seq[-1] = CosineConv2d(c3_in, self.nc, temperature=temperature)

    @staticmethod
    def _get_last_conv_in_channels(seq: nn.Sequential) -> int:
        """Get the input channels of the last Conv2d in a Sequential."""
        last = seq[-1]
        if isinstance(last, nn.Conv2d):
            return last.in_channels
        raise TypeError(f"Expected last layer to be Conv2d, got {type(last)}")

    def bias_init(self) -> None:
        """Initialize biases with standard YOLO logit prior.

        Box branch: bias → 1.0 (same as parent).
        Cls branch (CosineConv2d): bias uses the standard YOLO formula
        so training loss starts in the same range as vanilla Detect.
        """
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            last_cls = b[-1]
            if isinstance(last_cls, CosineConv2d):
                last_cls.bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)
            else:
                last_cls.bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)

    def _get_cosine_biases(self) -> torch.Tensor:
        """Collect learnable biases from all CosineConv2d layers.

        Returns a (1, nc, 1) tensor with the mean bias across scales.
        """
        biases = []
        for i in range(self.nl):
            last_cls = self.cv3[i][-1]
            if isinstance(last_cls, CosineConv2d):
                biases.append(last_cls.bias.detach())
        if not biases:
            return torch.tensor(0.0)
        # Average biases across scales (they are per-class, same shape)
        return torch.stack(biases).mean(dim=0).view(1, -1, 1)

    def _inference(self, x):
        """Override parent to debias cosine scores before sigmoid.

        Training: standard biased logits (loss behaviour unchanged).
        Inference: debiased scores so confidence = sigmoid(scale * cos),
        which maps cos=0 → conf≈0.5, cos>0 → conf>0.5, providing
        interpretable probability estimates in cosine space.
        """
        # Call parent _inference first (does box decoding + sigmoid)
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            from ultralytics.utils.tal import make_anchors
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        # Debiase: remove per-class bias before sigmoid
        # cls = scale * cos + bias  →  cls_debiased = scale * cos = cls - bias
        bias = self._get_cosine_biases().to(device=cls.device, dtype=cls.dtype)
        cls_debiased = cls - bias
        return torch.cat((dbox, cls_debiased.sigmoid()), 1)
