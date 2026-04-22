"""Tests for fsod.modules.dual_path_fusion."""
from __future__ import annotations

import pytest
import torch

from fsod.modules.dual_path_fusion import fuse_box_scores


def _toy_inputs(n: int = 4, c: int = 3, device: str = "cpu"):
    torch.manual_seed(0)
    s_yolo = torch.tensor([0.9, 0.5, 0.2, 0.7], device=device)
    yolo_cls = torch.tensor([0, 1, 2, 0], dtype=torch.long, device=device)
    # s_vlm: row 0 strongly supports class 0, row 1 supports class 2 (mismatch),
    # row 2 supports class 2 (match), row 3 supports class 0 (match).
    s_vlm = torch.tensor(
        [
            [0.30, 0.05, -0.10],
            [-0.05, -0.05, 0.30],
            [-0.10, 0.05, 0.25],
            [0.28, 0.04, -0.05],
        ],
        device=device,
    )
    return s_yolo[:n], yolo_cls[:n], s_vlm[:n]


def test_fixed_gamma_zero_reduces_to_yolo():
    s_yolo, cls, s_vlm = _toy_inputs()
    fused = fuse_box_scores(s_yolo, cls, s_vlm, mode="fixed", gamma=0.0)
    torch.testing.assert_close(fused, s_yolo, rtol=1e-5, atol=1e-6)


def test_fixed_gamma_one_uses_vlm_only():
    s_yolo, cls, s_vlm = _toy_inputs()
    fused = fuse_box_scores(s_yolo, cls, s_vlm, mode="fixed", gamma=1.0,
                            vlm_temperature=100.0)
    # With γ=1, fused == s_vlm_for_class (post softmax).
    expected = torch.softmax(s_vlm * 100.0, dim=-1).gather(1, cls.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(fused.clamp(0, 1), expected.clamp(0, 1), rtol=1e-5, atol=1e-6)


def test_consistent_box_gets_boost_inconsistent_box_gets_penalty():
    """Box 0 (yolo says cls 0, vlm agrees) should rise above box 1 (mismatch)."""
    s_yolo, cls, s_vlm = _toy_inputs()
    fused = fuse_box_scores(s_yolo, cls, s_vlm, mode="multiplicative",
                            vlm_temperature=100.0)
    # Box 0: yolo 0.9 × p(vlm=cls0) ≈ 0.9 * ~1.0 (sharp softmax peaked at cls 0)
    # Box 1: yolo 0.5 × p(vlm=cls1) where vlm prefers cls 2 → near 0.
    assert fused[0] > fused[1]


def test_output_in_unit_range():
    s_yolo, cls, s_vlm = _toy_inputs()
    for mode in ("fixed", "multiplicative", "rerank"):
        fused = fuse_box_scores(s_yolo, cls, s_vlm, mode=mode, gamma=0.5)
        assert torch.all(fused >= 0.0) and torch.all(fused <= 1.0)


def test_empty_input_returns_empty():
    s_yolo = torch.empty(0)
    cls = torch.empty(0, dtype=torch.long)
    s_vlm = torch.empty(0, 5)
    fused = fuse_box_scores(s_yolo, cls, s_vlm, mode="fixed", gamma=0.5)
    assert fused.shape == (0,)


def test_shape_mismatch_raises():
    s_yolo = torch.tensor([0.5, 0.6])
    cls = torch.tensor([0, 1], dtype=torch.long)
    s_vlm = torch.zeros(3, 4)  # wrong N
    with pytest.raises(ValueError):
        fuse_box_scores(s_yolo, cls, s_vlm, mode="fixed")


def test_invalid_mode_raises():
    s_yolo, cls, s_vlm = _toy_inputs()
    with pytest.raises(ValueError):
        fuse_box_scores(s_yolo, cls, s_vlm, mode="bogus")  # type: ignore[arg-type]


def test_invalid_gamma_raises():
    s_yolo, cls, s_vlm = _toy_inputs()
    with pytest.raises(ValueError):
        fuse_box_scores(s_yolo, cls, s_vlm, mode="fixed", gamma=1.5)
