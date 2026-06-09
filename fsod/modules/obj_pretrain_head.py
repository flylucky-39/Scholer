"""FSODDetectWithObjPretrain: YOLO head with cosine classifier + class-agnostic objectness.

Architecture per scale: [box(4*reg_max), cls(nc), obj(1)]

Base pretraining:
  - Train all parameters (cosine classifier + obj_pred) on base classes
  - Loss = box_loss + cls_loss + dfl_loss + obj_loss
  - Produces a strong objectness prior from abundant base-class data

Finetuning:
  - Load base weights (obj_pred transferred, cls re-initialized via prototypes)
  - Freeze obj_pred (objectness knowledge from base training is preserved)
  - Initialize cosine classifier with class prototypes
  - Inference: score = sigmoid(obj) * sigmoid(cls)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from fsod.modules.cosine_head import CosineConv2d


class FSODDetectWithObjPretrain(Detect):
    """YOLO Detect head with Cosine Classifier + class-agnostic objectness.

    For each scale, outputs [box(4*reg_max), cls(nc), obj(1)].
    The objectness branch is designed to be:
      - Trained on abundant base classes → strong foreground detector
      - Frozen during few-shot finetuning → preserves objectness knowledge
    """

    def __init__(self, nc: int = 80, ch: tuple = (), temperature: float = 5.0) -> None:
        super().__init__(nc=nc, ch=ch)
        self.temperature = temperature

        # Intermediate channel count (same compute as cv3 internal channels)
        c3 = max(ch[0], min(self.nc, 100))

        # Replace cv3 last layer with CosineConv2d (same as FSODDetect)
        for i in range(self.nl):
            old_seq = self.cv3[i]
            c3_in = old_seq[-1].in_channels
            old_seq[-1] = CosineConv2d(c3_in, self.nc, temperature=temperature)

        # Objectness branch: [Conv(ch→c3,3), Conv(c3→c3,1), Conv2d(c3→1,1)]
        self.obj_pred = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 1),
                nn.Conv2d(c3, 1, 1),
            )
            for x in ch
        )

        # Total outputs per anchor: box + class + objectness
        self.no = self.reg_max * 4 + self.nc + 1

    def forward(self, x):
        """Forward: [box(4*reg_max), cls(nc), obj(1)] per scale."""
        for i in range(self.nl):
            x[i] = torch.cat(
                (self.cv2[i](x[i]),    # box: 4*reg_max
                 self.cv3[i](x[i]),    # cls: nc
                 self.obj_pred[i](x[i])),  # obj: 1
                dim=1,
            )
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x):
        """Decode boxes, weight class scores by objectness."""
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # 3-way split: box | cls | obj
        box, cls_obj, obj = x_cat.split((self.reg_max * 4, self.nc, 1), 1)

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        # confidence = sigmoid(objectness) * sigmoid(class_score)
        cls_weighted = cls_obj.sigmoid() * obj.sigmoid()

        return torch.cat((dbox, cls_weighted), 1)

    def bias_init(self):
        """Initialize biases: box=1.0, cls=log(5/nc/(640/s)^2), obj=log(0.1/0.9)."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)  # cls
        for obj_i in self.obj_pred:
            obj_i[-1].bias.data.fill_(math.log(0.1 / 0.9))  # obj: sigmoid ≈ 0.1

    def freeze_obj_pred(self):
        """Freeze objectness branch — preserves base-trained objectness during finetune."""
        for p in self.obj_pred.parameters():
            p.requires_grad_(False)

    def unfreeze_obj_pred(self):
        """Unfreeze objectness branch for full-network training."""
        for p in self.obj_pred.parameters():
            p.requires_grad_(True)


class FSODObjPretrainLoss(v8DetectionLoss):
    """v8DetectionLoss + class-agnostic objectness BCE loss.

    Splits head output [box(4*reg_max), cls(nc), obj(1)] and adds binary
    objectness loss using the foreground mask from TAL.
    """

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk)
        # Override no: box(4*reg_max) + cls(nc) + obj(1)
        m = model.model[-1]
        self.no = m.reg_max * 4 + m.nc + 1

    def __call__(self, preds, batch):
        """Return (loss_sum * batch_size, loss_items_detached) with 4 items: [box, cls, dfl, obj]."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, obj

        feats = preds[1] if isinstance(preds, tuple) else preds

        # 3-way split: box(4*reg_max) + cls(nc) + obj(1)
        all_preds = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        pred_distri, pred_scores, pred_obj = all_preds.split((self.reg_max * 4, self.nc, 1), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_obj = pred_obj.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Predicted boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Task-aligned assigner
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # --- Objectness loss (BCE) ---
        obj_target = fg_mask.unsqueeze(-1).to(dtype=dtype)
        loss[3] = self.bce(pred_obj, obj_target).sum() / target_scores_sum

        # --- Class loss (BCE) ---
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # --- Box + DFL loss ---
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.hyp.cls  # reuse cls gain for objectness

        return loss.sum() * batch_size, loss.detach()

# ---------------------------------------------------------------------------
# Monkey-patch: register FSODDetectWithObjPretrain in ultralytics nn.tasks so
# parse_model can find it via globals() and handle its channel args correctly.
# ---------------------------------------------------------------------------
import ultralytics.nn.tasks as _tasks

_tasks.FSODDetectWithObjPretrain = FSODDetectWithObjPretrain
# Also alias FSODDetect for the identity check in parse_model channel handling
_tasks.FSODDetect = FSODDetectWithObjPretrain

# ---------------------------------------------------------------------------
# Patch DetectionModel.init_criterion: return FSODObjPretrainLoss when the head
# is FSODDetectWithObjPretrain. This is at the CLASS level so it survives EMA
# deepcopy (trainer.ema.ema) and DDP wrapping — the EMA copy inherits the
# patched method from its class.
# ---------------------------------------------------------------------------
_original_init_criterion = _tasks.DetectionModel.init_criterion


def _patched_init_criterion(self):
    if hasattr(self, "model") and len(self.model) > 0:
        head = self.model[-1]
        if isinstance(head, FSODDetectWithObjPretrain):
            return FSODObjPretrainLoss(self)
    return _original_init_criterion(self)


_tasks.DetectionModel.init_criterion = _patched_init_criterion
