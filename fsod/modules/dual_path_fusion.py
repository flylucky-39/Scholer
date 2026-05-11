"""Score fusion for Direction C dual-path inference.

Combines per-box YOLO confidence with CLIP-verifier per-class similarity. The
verifier returns raw cosine similarities in [-1, 1] over all C classes; we
first turn that into a per-box probability via temperature-scaled softmax,
then fuse with the YOLO confidence under one of three modes.

This module is deliberately framework-agnostic. It takes pure tensors and
returns tensors; the caller (``eval_fsod_dualpath.py``) handles I/O, NMS,
and metric accumulation.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

FusionMode = Literal["fixed", "multiplicative", "rerank"]


def fuse_box_scores(
    s_yolo: Tensor,
    yolo_cls: Tensor,
    s_vlm: Tensor,
    *,
    mode: FusionMode = "fixed",
    gamma: float = 0.5,
    vlm_temperature: float = 100.0,
    adaptive: bool = True,
) -> Tensor:
    """Fuse per-box YOLO confidence with VLM verifier similarity.

    Parameters
    ----------
    s_yolo : Tensor
        ``[N]`` post-NMS YOLO confidence (the max-class probability that
        survived NMS), already in [0, 1].
    yolo_cls : Tensor
        ``[N]`` integer class indices predicted by YOLO for each box.
    s_vlm : Tensor
        ``[N, C]`` raw cosine similarity from ``CLIPVerifier.verify`` in
        roughly [-1, 1]. Class index dim ``C`` must align with YOLO's.
    mode : {"fixed", "multiplicative", "rerank"}
        Fusion strategy:
          * ``"fixed"`` — convex combination ``(1-γ)·s_yolo + γ·s_vlm_for_class``
          * ``"multiplicative"`` — ``s_yolo · s_vlm_for_class`` (VLM as gate)
          * ``"rerank"`` — drop YOLO and use VLM only (diagnostic).
    gamma : float
        Mixing weight for ``"fixed"``. Ignored otherwise.
    adaptive : bool
        When True (default), gamma is scaled per-box by ``(1 - s_yolo)`` so
        that high-confidence boxes get less VLM influence. Set to False for
        the original global-gamma behaviour.
    vlm_temperature : float
        Logit temperature applied before softmax. CLIP's typical scale is
        ~100 (matches its ``logit_scale.exp()``).

    Returns
    -------
    Tensor
        ``[N]`` fused confidence in [0, 1].

    Raises
    ------
    ValueError
        On shape or mode mismatch.
    """
    if s_yolo.dim() != 1:
        raise ValueError(f"s_yolo must be [N], got {tuple(s_yolo.shape)}")
    if yolo_cls.dim() != 1 or yolo_cls.shape[0] != s_yolo.shape[0]:
        raise ValueError(
            f"yolo_cls shape {tuple(yolo_cls.shape)} must match s_yolo "
            f"{tuple(s_yolo.shape)}"
        )
    if s_vlm.dim() != 2 or s_vlm.shape[0] != s_yolo.shape[0]:
        raise ValueError(
            f"s_vlm shape {tuple(s_vlm.shape)} must be [N, C] with N matching "
            f"s_yolo {tuple(s_yolo.shape)}"
        )

    if s_yolo.numel() == 0:
        return s_yolo.clone()

    # Normalize VLM similarity into a per-class probability (per box) then
    # gather the column for YOLO's predicted class.
    s_vlm_prob = torch.softmax(s_vlm * vlm_temperature, dim=-1)
    cls_index = yolo_cls.long().to(s_vlm_prob.device).clamp(0, s_vlm_prob.shape[1] - 1)
    s_vlm_for_cls = s_vlm_prob.gather(1, cls_index.unsqueeze(1)).squeeze(1)
    s_vlm_for_cls = s_vlm_for_cls.to(s_yolo.device)

    # Per-box adaptive gamma: high YOLO confidence → less VLM influence.
    if adaptive and mode != "rerank":
        gamma_i = gamma * (1.0 - s_yolo)
    else:
        gamma_i = gamma

    if mode == "fixed":
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        fused = (1.0 - gamma_i) * s_yolo + gamma_i * s_vlm_for_cls
    elif mode == "multiplicative":
        if adaptive:
            # Interpolate between pure YOLO and multiplicative:
            # gamma_i=0 (high YOLO conf) → s_yolo;  gamma_i=1 → s_yolo * s_vlm
            fused_mul = s_yolo * s_vlm_for_cls
            fused = (1.0 - gamma_i) * s_yolo + gamma_i * fused_mul
        else:
            fused = s_yolo * s_vlm_for_cls
    elif mode == "rerank":
        fused = s_vlm_for_cls
    else:
        raise ValueError(f"unknown fusion mode: {mode!r}")

    return fused.clamp(0.0, 1.0)
