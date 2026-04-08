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
        score_c = s * cos(feature, weight_c)
    where s is a learnable temperature (scale) parameter.

    This is equivalent to L2-normalizing both the weight and the input
    before a standard 1x1 convolution, then scaling by temperature.
    """

    def __init__(self, in_channels: int, out_channels: int, temperature: float = 20.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        # Learnable temperature (log-space for positivity)
        self.scale = nn.Parameter(torch.tensor(math.log(temperature)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            (B, C_out, H, W) cosine similarity scores scaled by temperature
        """
        # Normalize weight: (C_out, C_in)
        w_norm = F.normalize(self.weight, dim=1)
        # Normalize input along channel dim: (B, C_in, H, W)
        x_norm = F.normalize(x, dim=1)
        # 1x1 conv with normalized weight → cosine similarity
        cos_sim = F.conv2d(x_norm, w_norm.unsqueeze(-1).unsqueeze(-1))
        # Scale by temperature
        return cos_sim * self.scale.exp()


class FSODDetect(Detect):
    """YOLO Detect head with Cosine Classifier replacing the final classification layer.

    Inherits from Detect and only modifies cv3 (classification branch):
    - The last nn.Conv2d(c3, nc, 1) is replaced with CosineConv2d(c3, nc).
    - Everything else (bbox regression, DFL, anchors, strides) is unchanged.
    """

    def __init__(self, nc: int = 80, ch: tuple = (), temperature: float = 20.0) -> None:
        super().__init__(nc=nc, ch=ch)
        self.temperature = temperature
        # Replace the last layer of each cv3 branch with CosineConv2d
        for i in range(self.nl):
            old_seq = self.cv3[i]
            # The last element is nn.Conv2d(c3, nc, 1)
            c3_in = self._get_last_conv_in_channels(old_seq)
            # Replace last layer
            old_seq[-1] = CosineConv2d(c3_in, self.nc, temperature=temperature)

    @staticmethod
    def _get_last_conv_in_channels(seq: nn.Sequential) -> int:
        """Get the input channels of the last Conv2d in a Sequential."""
        last = seq[-1]
        if isinstance(last, nn.Conv2d):
            return last.in_channels
        raise TypeError(f"Expected last layer to be Conv2d, got {type(last)}")

    def bias_init(self) -> None:
        """Initialize biases. Box branch uses parent logic; cls branch has no bias (cosine)."""
        for a, s in zip(self.cv2, self.stride):
            a[-1].bias.data[:] = 1.0  # box
        # CosineConv2d has no bias; skip cv3 bias init
