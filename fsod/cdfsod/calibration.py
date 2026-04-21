"""Domain-Adaptive Fusion (DAF) and Cross-Domain Prototype Calibration (CDPC).

DAF
    α(g) = sigmoid(k · (g − g0))
    Only two scalar parameters (k, g0). When trained jointly with the cosine
    head, α is bounded in (0, 1) and grows monotonically with the domain gap.

CDPC
    Pulls each visual prototype toward the corresponding text prototype,
    with the strength governed by α(g):

        P_v ← P_v + λ · α(g) · (P_t − P_v) / ||P_t − P_v||₂

    λ is a fixed step size (default 0.1) so the calibration never flips the
    prototype past the text anchor.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Domain-Adaptive Fusion
# ---------------------------------------------------------------------------

def alpha_from_gap(g: float, k: float = 6.0, g0: float = 0.35) -> float:
    """Map domain gap g ∈ [0, 1] to fusion weight α ∈ (0, 1).

    Defaults are chosen so that:
      * g = 0.20 (near in-domain) → α ≈ 0.28  (visual proto dominates)
      * g = 0.35 (mild shift)     → α = 0.50  (balanced)
      * g = 0.55 (large shift)    → α ≈ 0.77  (text proto dominates)
    """
    g = max(0.0, min(1.0, float(g)))
    return 1.0 / (1.0 + math.exp(-k * (g - g0)))


# ---------------------------------------------------------------------------
# Cross-Domain Prototype Calibration
# ---------------------------------------------------------------------------

def calibrate_prototypes(
    visual: Mapping[str, torch.Tensor],
    textual: Mapping[str, torch.Tensor],
    alpha: float,
    step: float = 0.1,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Pull each visual prototype toward its text counterpart.

    The two prototype sets must share the same vector dimension. Classes
    missing from either dictionary are passed through unchanged from
    ``visual``.

    Returns a NEW dict (input ``visual`` is not mutated).
    """
    out: dict[str, torch.Tensor] = {}
    alpha = max(0.0, min(1.0, float(alpha)))
    step = max(0.0, float(step))

    for name, v in visual.items():
        if name not in textual:
            out[name] = v.clone()
            continue

        t = textual[name].to(device=v.device, dtype=v.dtype)
        if t.shape != v.shape:
            raise ValueError(
                f"Prototype shape mismatch for '{name}': visual={tuple(v.shape)} text={tuple(t.shape)}"
            )

        diff = t - v
        norm = diff.norm().clamp_min(eps)
        unit = diff / norm

        out[name] = v + step * alpha * unit * norm.clamp_max(1.0)
        # We multiply by ``norm.clamp_max(1.0)`` so that already-close pairs do
        # not get over-perturbed: when the original visual prototype is already
        # near the text anchor, the calibration step shrinks naturally.

    return out


# ---------------------------------------------------------------------------
# Convex fusion (used as the actual init weight written to CosineConv2d)
# ---------------------------------------------------------------------------

def fuse_prototypes(
    visual: Mapping[str, torch.Tensor],
    textual: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Convex blend: P = (1-α) · normalize(P_v) + α · normalize(P_t), then renormalise.

    This is the actual prototype written into CosineConv2d.weight after CDPC
    has been applied to the visual side. We keep it as a separate function so
    that ablations can disable CDPC while still using DAF, or vice-versa.
    """
    out: dict[str, torch.Tensor] = {}
    alpha = max(0.0, min(1.0, float(alpha)))

    for name, v in visual.items():
        v_n = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
        if name not in textual:
            out[name] = v_n
            continue
        t = textual[name].to(device=v.device, dtype=v.dtype)
        t_n = F.normalize(t.unsqueeze(0), dim=1).squeeze(0)
        fused = (1.0 - alpha) * v_n + alpha * t_n
        out[name] = F.normalize(fused.unsqueeze(0), dim=1).squeeze(0)

    return out
