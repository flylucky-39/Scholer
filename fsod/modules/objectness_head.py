"""FSODDetectWithObjectness: YOLO detection head with cosine classifier + class-agnostic objectness.

Architecture per scale: [box_reg(4*reg_max), objectness(1), class_scores(nc)]

Training:
  - Objectness branch learns "is there an object?" from ALL objects (base + novel)
  - Classifier only needs to distinguish between object classes
  - Decouples detection from classification — critical for few-shot

Inference:
  confidence = sigmoid(objectness) × sigmoid(class_score)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import make_anchors

from fsod.modules.cosine_head import CosineConv2d


class FSODDetectWithObjectness(Detect):
    """YOLO Detect head with Cosine Classifier + class-agnostic objectness.

    For each scale, outputs [box(4*reg_max), obj(1), cls(nc)] concatenated.
    Objectness acts as a foreground filter, decoupling detection from classification.
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
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 1),
                nn.Conv2d(c3, 1, 1),
            )
            for x in ch
        )

        # Total outputs per anchor: box + objectness + class
        self.no = self.reg_max * 4 + 1 + self.nc

    def forward(self, x):
        """Forward: [box(reg_max*4), obj(1), cls(nc)] per scale."""
        for i in range(self.nl):
            x[i] = torch.cat(
                (self.cv2[i](x[i]),   # box: 4*reg_max
                 self.cv4[i](x[i]),   # obj: 1
                 self.cv3[i](x[i])),  # cls: nc
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

        # 3-way split: box | obj | cls
        box, obj, cls = x_cat.split((self.reg_max * 4, 1, self.nc), 1)

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        # confidence = sigmoid(objectness) × sigmoid(class_score)
        cls_weighted = cls.sigmoid() * obj.sigmoid()

        return torch.cat((dbox, cls_weighted), 1)

    def bias_init(self):
        """Initialize biases: box=1.0, cls=log(5/nc/(640/s)^2), obj=log(0.1/0.9)."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)  # cls
        for cv4_i in self.cv4:
            cv4_i[-1].bias.data.fill_(math.log(0.1 / 0.9))  # obj: sigmoid ≈ 0.1

    def set_background_proto(self, bg_proto, gamma: float = 0.3):
        """Inject background prototype into all CosineConv2d layers.

        Args:
            bg_proto: Background prototype tensor of shape (c3,).
            gamma: Background suppression strength.
        """
        for i in range(self.nl):
            cosine_layer = self.cv3[i][-1]
            if isinstance(cosine_layer, CosineConv2d):
                cosine_layer.set_background_proto(bg_proto, gamma)
        print(f"  Injected background prototype into {self.nl} scales (gamma={gamma})")


class FSODObjectnessLoss(v8DetectionLoss):
    """v8DetectionLoss + class-agnostic objectness BCE loss.

    The loss splits [box(reg_max*4), obj(1), cls(nc)] from the head output
    and adds a binary objectness loss using the foreground mask from TAL.
    """

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk)
        # Override no to include objectness channel (head outputs reg_max*4+1+nc)
        m = model.model[-1]
        self.no = m.reg_max * 4 + 1 + m.nc

    def __call__(self, preds, batch):
        """Return (loss_sum * batch_size, loss_items_detached) with 4 items: [box, cls, dfl, obj]."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, obj

        feats = preds[1] if isinstance(preds, tuple) else preds

        # 3-way split: box(reg_max*4) + obj(1) + cls(nc)
        all_preds = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        pred_distri, pred_obj, pred_scores = all_preds.split((self.reg_max * 4, 1, self.nc), 1)

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
        # fg_mask indicates anchors assigned to GT objects → objectness target
        obj_target = fg_mask.unsqueeze(-1).to(dtype=dtype)
        loss[3] = self.bce(pred_obj, obj_target).sum() / target_scores_sum

        # --- Class loss (BCE, same as original) ---
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


def add_objectness_callback(model):
    """Register callbacks on a YOLO model to use FSODDetectWithObjectness loss.

    Call this BEFORE model.train() to inject the custom criterion.
    """
    def _on_pretrain_routine_end(trainer):
        if not isinstance(trainer.model.model[-1], FSODDetectWithObjectness):
            return  # skip if head doesn't have objectness
        trainer.model.criterion = FSODObjectnessLoss(trainer.model)
        # Update loss names to include obj_loss
        trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss", "obj_loss")

    model.add_callback("on_pretrain_routine_end", _on_pretrain_routine_end)


# ---------------------------------------------------------------------------
# Monkey-patch: register FSODDetectWithObjectness in ultralytics nn.tasks so
# parse_model can find it via globals() and handle its channel args correctly.
# ---------------------------------------------------------------------------
import ultralytics.nn.tasks as _tasks

_tasks.FSODDetectWithObjectness = FSODDetectWithObjectness

# ---------------------------------------------------------------------------
# Patch DetectionModel.init_criterion: return FSODObjectnessLoss when the head
# is FSODDetectWithObjectness. This is needed for both training AND validation
# (the validator deepcopies the EMA model, which also needs the correct loss).
# ---------------------------------------------------------------------------
_original_init_criterion = _tasks.DetectionModel.init_criterion


def _patched_init_criterion(self):
    if hasattr(self, "model") and len(self.model) > 0:
        head = self.model[-1]
        if isinstance(head, FSODDetectWithObjectness):
            return FSODObjectnessLoss(self)
    return _original_init_criterion(self)


_tasks.DetectionModel.init_criterion = _patched_init_criterion
