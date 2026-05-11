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
    # With gamma=1, fused == s_vlm_for_class (post softmax).
    expected = torch.softmax(s_vlm * 100.0, dim=-1).gather(1, cls.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(fused.clamp(0, 1), expected.clamp(0, 1), rtol=1e-5, atol=1e-6)


def test_consistent_box_gets_boost_inconsistent_box_gets_penalty():
    """Box 0 (yolo says cls 0, vlm agrees) should rise above box 1 (mismatch)."""
    s_yolo, cls, s_vlm = _toy_inputs()
    fused = fuse_box_scores(s_yolo, cls, s_vlm, mode="multiplicative",
                            vlm_temperature=100.0)
    # Box 0: yolo 0.9 * p(vlm=cls0) ~ 0.9 * ~1.0 (sharp softmax peaked at cls 0)
    # Box 1: yolo 0.5 * p(vlm=cls1) where vlm prefers cls 2 -> near 0.
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


# ------------------------------------------------------------------
# New tests for adaptive gamma
# ------------------------------------------------------------------

def test_adaptive_disabled_matches_original():
    """adaptive=False should produce identical results to the original behaviour."""
    s_yolo, cls, s_vlm = _toy_inputs()
    fused_new = fuse_box_scores(s_yolo, cls, s_vlm, mode="fixed", gamma=0.5,
                                adaptive=False)
    # The "old" path with default (adaptive=True) + gamma_i calculation produces a
    # different result. With adaptive=False the gamma_i is just gamma (scalar 0.5).
    expected = (1.0 - 0.5) * s_yolo + 0.5 * torch.softmax(s_vlm * 100.0, dim=-1).gather(
        1, cls.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(fused_new, expected.clamp(0, 1), rtol=1e-5, atol=1e-6)


def test_adaptive_disabled_multiplicative_matches_original():
    """adaptive=False multiplicative should match original s_yolo * s_vlm."""
    s_yolo, cls, s_vlm = _toy_inputs()
    fused = fuse_box_scores(s_yolo, cls, s_vlm, mode="multiplicative",
                            vlm_temperature=100.0, adaptive=False)
    s_vlm_prob = torch.softmax(s_vlm * 100.0, dim=-1)
    s_vlm_cls = s_vlm_prob.gather(1, cls.unsqueeze(1)).squeeze(1)
    expected = s_yolo * s_vlm_cls
    torch.testing.assert_close(fused, expected.clamp(0, 1), rtol=1e-5, atol=1e-6)


def test_adaptive_reduces_vlm_influence_on_high_conf():
    """When s_yolo is high, adaptive gamma should reduce VLM influence vs non-adaptive."""
    s_yolo = torch.tensor([0.9, 0.95, 0.99])
    cls = torch.tensor([0, 0, 0], dtype=torch.long)
    s_vlm = torch.tensor([[0.1, -0.5, -0.5], [0.2, -0.5, -0.5], [0.3, -0.5, -0.5]])
    # VLM disagrees with YOLO (low similarity for class 0).
    # With adaptive=True, high-conf boxes should keep scores close to s_yolo.
    fused_adaptive = fuse_box_scores(s_yolo, cls, s_vlm, mode="fixed", gamma=0.5,
                                     vlm_temperature=100.0, adaptive=True)
    fused_global = fuse_box_scores(s_yolo, cls, s_vlm, mode="fixed", gamma=0.5,
                                   vlm_temperature=100.0, adaptive=False)
    # Adaptive fused scores should be closer to s_yolo than global fused scores.
    diff_adaptive = (fused_adaptive - s_yolo).abs().mean()
    diff_global = (fused_global - s_yolo).abs().mean()
    assert diff_adaptive < diff_global


def test_adaptive_preserves_vlm_help_on_low_conf():
    """When s_yolo is low, adaptive gamma should still allow VLM to influence."""
    s_yolo = torch.tensor([0.15, 0.20, 0.10])
    cls = torch.tensor([0, 0, 0], dtype=torch.long)
    s_vlm = torch.tensor([[0.8, -0.5, -0.5], [0.7, -0.5, -0.5], [0.9, -0.5, -0.5]])
    fused_adaptive = fuse_box_scores(s_yolo, cls, s_vlm, mode="fixed", gamma=0.5,
                                     vlm_temperature=100.0, adaptive=True)
    fused_global = fuse_box_scores(s_yolo, cls, s_vlm, mode="fixed", gamma=0.5,
                                   vlm_temperature=100.0, adaptive=False)
    # Both should boost above s_yolo since VLM agrees. Adaptive gamma_i ~= 0.5*0.85 ~= 0.425.
    # This is close enough to global gamma=0.5 that results should be similar.
    assert torch.all(fused_adaptive > s_yolo)
    assert torch.all(fused_global > s_yolo)


def test_adaptive_multiplicative_high_conf_near_yolo():
    """In multiplicative mode, high-conf boxes should stay near s_yolo with adaptive."""
    s_yolo = torch.tensor([0.92, 0.88])
    cls = torch.tensor([0, 1], dtype=torch.long)
    s_vlm = torch.tensor([[0.5, -0.5], [-0.5, 0.4]])
    fused = fuse_box_scores(s_yolo, cls, s_vlm, mode="multiplicative", gamma=0.5,
                            vlm_temperature=100.0, adaptive=True)
    # High YOLO conf -> gamma_i small -> fused close to s_yolo.
    assert torch.all(fused > 0.85)


def test_adaptive_rerank_ignored():
    """Rerank mode should ignore adaptive (VLM-only, no YOLO to base adaptation on)."""
    s_yolo, cls, s_vlm = _toy_inputs()
    fused_adapt = fuse_box_scores(s_yolo, cls, s_vlm, mode="rerank", adaptive=True)
    fused_global = fuse_box_scores(s_yolo, cls, s_vlm, mode="rerank", adaptive=False)
    torch.testing.assert_close(fused_adapt, fused_global)
