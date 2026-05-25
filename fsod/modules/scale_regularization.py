"""Scale regularization for CosineConv2d to suppress overconfidence in FSOD.

Extends ``v8DetectionLoss`` with an L2 penalty on the per-class temperature
(``scale.exp()``) in each ``CosineConv2d`` layer.  This prevents the classifier
from assigning arbitrarily high confidence scores, which is the root cause of
the "too many detection boxes" problem when P is low.

Usage in a training callback::

    from fsod.modules.scale_regularization import ScaleRegDetectionLoss

    def _patch_loss(trainer):
        trainer.criterion = ScaleRegDetectionLoss(
            trainer.model, reg_weight=0.01
        )
    model.add_callback("on_pretrain_routine_end", _patch_loss)
"""

from __future__ import annotations

import torch
from ultralytics.utils.loss import v8DetectionLoss


class ScaleRegDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with scale regularization for CosineConv2d classification heads.

    For each ``CosineConv2d`` in the detection head, we add::

        L_reg = reg_weight * mean(scale.exp())

    where ``scale`` is the per-class temperature stored in log-space.  This
    penalises large temperatures that cause the model to produce high-confidence
    predictions on background / negative anchors.

    Parameters
    ----------
    model : nn.Module
        Full YOLO model (same as passed to ``v8DetectionLoss``).
    reg_weight : float
        Strength of the scale regularisation term.  Start with 0.01–0.1.
    tal_topk : int
        Top-k for ``TaskAlignedAssigner`` (same as ``v8DetectionLoss``).
    """

    def __init__(
        self,
        model,
        reg_weight: float = 0.01,
        tal_topk: int = 10,
    ) -> None:
        super().__init__(model, tal_topk)
        self.reg_weight = reg_weight

    def __call__(self, preds, batch):
        """Original BCE loss + scale regularisation for CosineConv2d heads."""
        loss, loss_items = super().__call__(preds, batch)

        if self.reg_weight <= 0.0:
            return loss, loss_items

        detect = self.model.model[-1]
        if not hasattr(detect, "cv3"):
            return loss, loss_items

        # Iterate over all scale-specific classification branches (P3 / P4 / P5).
        reg = torch.tensor(0.0, device=loss.device)
        n = 0
        for cv3_scale in detect.cv3:
            cos_layer = cv3_scale[-1]
            if hasattr(cos_layer, "scale"):
                # scale is stored in log-space; exp() gives the actual temperature.
                reg = reg + cos_layer.scale.exp().mean()
                n += 1

        if n > 0:
            loss = loss + (reg / n) * self.reg_weight

        return loss, loss_items
