"""Learnable Background Suppression — a "learnable/addition" paradigm baseline.

Instead of fixed orthogonalisation (p' = p - β·⟨p,b̂⟩·b̂), this module uses a
learnable background query with cross-attention to extract the background
component from each class prototype.  The module is trained on base classes
and then applied to novel classes.

This serves as a controlled contrast to the fixed (non-learnable) background
suppression: if learnable suppression underperforms, it validates the
"subtraction beats addition in few-shot" hypothesis.

Mechanism:
  1. bg_query (learnable, initialised from extracted bg_proto) acts as Q
  2. Class prototype p is split into segments → K, V
  3. Cross-attention: bg_query attends to prototype segments
  4. FFN refines the attended background component
  5. p' = normalize(p - β · attended_background)

Module is trained on base classes to minimise cos(p'_i, b) — i.e., to make
the refined prototypes as orthogonal as possible to the background direction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Learnable Background Suppression Module
# ---------------------------------------------------------------------------

class LearnableBackgroundSuppression(nn.Module):
    """Learnable background suppression via cross-attention.

    Args:
        dim: Feature dimension (256 for yolo11s cv3 penultimate).
        num_segments: Number of segments to split the prototype into for
                      cross-attention.  The prototype (dim,) is reshaped to
                      (num_segments, dim//num_segments) and projected to dim.
        n_heads: Number of attention heads.
        ffn_expansion: FFN hidden dimension = dim * ffn_expansion.
        beta_init: Initial suppression strength.
    """

    def __init__(
        self,
        dim: int = 256,
        num_segments: int = 8,
        n_heads: int = 4,
        ffn_expansion: int = 4,
        beta_init: float = 0.5,
    ) -> None:
        super().__init__()
        assert dim % num_segments == 0, f"dim {dim} must be divisible by num_segments {num_segments}"

        self.dim = dim
        self.num_segments = num_segments
        self.seg_dim = dim // num_segments

        # Learnable background query — initialised from extracted bg_proto later
        self.bg_query = nn.Parameter(torch.zeros(1, 1, dim))

        # Project each segment from seg_dim → dim
        self.seg_proj = nn.Linear(self.seg_dim, dim)

        # Multi-head cross-attention: bg_query (Q) attends to segments (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim),
        )

        # Learnable suppression strength (sigmoid → [0, 1])
        self.beta_raw = nn.Parameter(torch.tensor(self._inv_sigmoid(beta_init)))

    @staticmethod
    def _inv_sigmoid(x: float) -> float:
        import math
        return math.log(x / (1 - x))

    def init_bg_query(self, bg_proto: torch.Tensor) -> None:
        """Initialise bg_query from the extracted background prototype.

        Args:
            bg_proto: Background prototype of shape (dim,).
        """
        self.bg_query.data.copy_(bg_proto.view(1, 1, self.dim).to(self.bg_query.device))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Apply learnable background suppression to a prototype.

        Args:
            p: Prototype tensor of shape (dim,) or (B, dim).

        Returns:
            Refined prototype of the same shape, L2-normalised.
        """
        single = p.dim() == 1
        if single:
            p = p.unsqueeze(0)  # (1, dim)

        B = p.shape[0]

        # 1. Split prototype into segments and project to dim
        segments = p.view(B, self.num_segments, self.seg_dim)          # (B, N, seg_dim)
        segments = self.seg_proj(segments)                              # (B, N, dim)

        # 2. Cross-attention: bg_query attends to prototype segments
        bg_q = self.bg_query.expand(B, -1, -1)                         # (B, 1, dim)
        attended, _ = self.cross_attn(query=bg_q, key=segments, value=segments)
        attended = attended.squeeze(1)                                  # (B, dim)

        # 3. FFN refinement
        attended = self.ffn(attended)                                   # (B, dim)

        # 4. Subtract background component
        beta = torch.sigmoid(self.beta_raw)
        p_refined = p - beta * attended

        # 5. L2-normalise
        p_refined = F.normalize(p_refined, dim=1)

        if single:
            p_refined = p_refined.squeeze(0)
        return p_refined


# ---------------------------------------------------------------------------
# 2. Training on Base Classes
# ---------------------------------------------------------------------------

def train_learnable_bg_suppression(
    module: LearnableBackgroundSuppression,
    base_prototypes: dict[str, torch.Tensor],
    bg_proto: torch.Tensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    epochs: int = 200,
    device: str = "cuda:0",
) -> LearnableBackgroundSuppression:
    """Train the learnable background suppression module on base classes.

    The module learns to make base-class prototypes orthogonal to the
    background direction.  Loss = mean(cos(p'_i, bg_unit)) across all
    base classes — we minimise this so the refined prototypes have
    minimal alignment with the background.

    Args:
        module: LearnableBackgroundSuppression instance.
        base_prototypes: Dict class_name → prototype tensor (dim,).
        bg_proto: Background prototype tensor (dim,).
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        epochs: Training epochs.
        device: Torch device.

    Returns:
        Trained module (in eval mode).
    """
    module = module.to(device)
    module.train()

    # Initialise bg_query from extracted bg_proto
    module.init_bg_query(bg_proto)

    # Build training tensors: stack all base class prototypes
    proto_list = []
    for cls_name in sorted(base_prototypes.keys()):
        proto = base_prototypes[cls_name]
        if proto.norm() == 0:
            continue
        proto_list.append(proto)
    X = torch.stack(proto_list).to(device)  # (N, dim)
    bg_unit = F.normalize(bg_proto, dim=0).to(device)  # (dim,)

    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"  Training learnable background suppression on {len(proto_list)} base classes "
          f"for {epochs} epochs (lr={lr}, wd={weight_decay})...")

    for epoch in range(epochs):
        p_refined = module(X)  # (N, dim)

        # Loss: mean cosine similarity between refined prototypes and background
        cos_sim = (p_refined * bg_unit.unsqueeze(0)).sum(dim=1)  # (N,)
        loss = cos_sim.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            beta_val = torch.sigmoid(module.beta_raw).item()
            print(f"    Epoch {epoch+1:4d}/{epochs}: loss={loss.item():.4f}, "
                  f"mean_cos_bg={cos_sim.mean().item():.4f}, beta={beta_val:.4f}")

    module.eval()
    return module


# ---------------------------------------------------------------------------
# 3. Application to Novel Class Prototypes
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_learnable_bg_suppression(
    module: LearnableBackgroundSuppression,
    prototypes: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply trained learnable background suppression to (novel) class prototypes.

    Args:
        module: Trained LearnableBackgroundSuppression.
        prototypes: Dict class_name → prototype tensor (dim,).

    Returns:
        Dict class_name → refined prototype tensor (dim,).
    """
    module.eval()
    refined: dict[str, torch.Tensor] = {}

    for cls_name, proto in prototypes.items():
        if proto.norm() == 0:
            refined[cls_name] = proto
            continue
        refined[cls_name] = module(proto)

    # Report statistics
    if refined:
        cos_before = []
        cos_after = []
        bg_query = F.normalize(module.bg_query.squeeze(), dim=0)
        for cls_name in prototypes:
            if prototypes[cls_name].norm() == 0:
                continue
            cos_before.append(
                torch.dot(F.normalize(prototypes[cls_name], dim=0), bg_query).item()
            )
            cos_after.append(
                torch.dot(refined[cls_name], bg_query).item()
            )
        import numpy as np
        beta_val = torch.sigmoid(module.beta_raw).item()
        print(f"  Learnable BG suppression (beta={beta_val:.4f}): "
              f"cos_bg {np.mean(cos_before):.4f} → {np.mean(cos_after):.4f} "
              f"(avg reduction: {np.mean(cos_before) - np.mean(cos_after):.4f})")

    return refined