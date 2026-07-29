"""Shot-Adaptive Background Suppression.

Instead of per-class cos_bg (which doesn't correlate with shot count),
use a simple shot-based schedule:

    beta = beta_max * exp(-k * (shot - 1))

Where:
  - shot=1: beta = beta_max (full suppression, proven effective)
  - shot=3: beta ≈ beta_max * 0.5 (reduced)
  - shot=5: beta ≈ beta_max * 0.1 (minimal)
  - shot=10: beta ≈ 0 (disabled)

The same decay applies to the inference-time gamma.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def shot_adaptive_params(
    shot: int,
    beta_max: float = 0.5,
    gamma_max: float = 0.3,
    decay_rate: float = 0.5,
) -> tuple[float, float]:
    """Compute shot-adaptive beta and gamma.

    Uses exponential decay:
        beta  = beta_max  * exp(-decay_rate * (shot - 1))
        gamma = gamma_max * exp(-decay_rate * (shot - 1))

    Args:
        shot: Number of support instances per class.
        beta_max: Maximum beta at shot=1.
        gamma_max: Maximum gamma at shot=1.
        decay_rate: Controls how fast suppression decays with more shots.

    Returns:
        (beta, gamma) tuple.

    Shot → params table (decay_rate=0.5):
        shot=1:  beta=0.500, gamma=0.300  (full)
        shot=2:  beta=0.237, gamma=0.142
        shot=3:  beta=0.113, gamma=0.068  (reduced)
        shot=5:  beta=0.025, gamma=0.015  (minimal)
        shot=10: beta=0.001, gamma=0.001  (virtually off)
    """
    factor = math.exp(-decay_rate * (shot - 1))
    beta = beta_max * factor
    gamma = gamma_max * factor
    print(f"  Shot-adaptive (shot={shot}): beta={beta:.4f}, gamma={gamma:.4f}")
    return beta, gamma


def refine_prototypes_shot_adaptive(
    prototypes: dict[str, torch.Tensor],
    bg_proto: torch.Tensor,
    shot: int,
    beta_max: float = 0.5,
    decay_rate: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Refine prototypes with shot-adaptive beta.

    Args:
        prototypes: Dict class_name → prototype (D,).
        bg_proto: Background prototype (D,).
        shot: Number of support instances per class.
        beta_max: Maximum beta at shot=1.
        decay_rate: Decay rate for shot→beta mapping.

    Returns:
        Refined prototypes.
    """
    if bg_proto.norm() == 0:
        return prototypes

    beta, _ = shot_adaptive_params(shot, beta_max=beta_max, decay_rate=decay_rate)

    bg_norm = F.normalize(bg_proto, dim=0)
    device = next(iter(prototypes.values())).device
    bg_norm = bg_norm.to(device)

    refined = {}
    for cls_name, proto in prototypes.items():
        if proto.norm() == 0:
            refined[cls_name] = proto
            continue
        proj = torch.dot(proto, bg_norm) * bg_norm
        refined[cls_name] = F.normalize(proto - beta * proj, dim=0)

    import numpy as np
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
        print(f"  Shot-adaptive refinement (beta={beta:.4f}): "
              f"cos_bg {np.mean(cos_before):.4f} → {np.mean(cos_after):.4f}")

    return refined