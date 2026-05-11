"""Train YOLO-FSOD with multi-template Florence-2 description aggregation.

Extends train_fsod.py with --num-templates N support for prompt template
quantity ablation. Uses multiple Florence-2 caption tasks per crop and
aggregates their embeddings before FiLM training.

Usage:
  # Single template (DETAILED_CAPTION, same as baseline fused)
  python scripts/train_fsod_multitemplate.py \
    --config configs/baseline_voc_10shot.yaml --stage finetune \
    --prototype --florence2 ~/epfs/07_FSOD_LLM/models/Florence-2-base/ \
    --num-templates 1

  # Multi-template ablation
  python scripts/train_fsod_multitemplate.py \
    --config configs/baseline_voc_10shot.yaml --stage finetune \
    --prototype --florence2 ~/epfs/07_FSOD_LLM/models/Florence-2-base/ \
    --num-templates 3 --run-suffix templates3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401


# ---------------------------------------------------------------------------
# Florence-2 caption task prompts (ordered by detail level)
# ---------------------------------------------------------------------------
CAPTION_TASKS = [
    "<DETAILED_CAPTION>",
    "<CAPTION>",
    "<MORE_DETAILED_CAPTION>",
]


# ---------------------------------------------------------------------------
# Multi-template description generation (replaces generate_and_encode_descriptions)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_multitemplate_descriptions(
    model_path: str | Path,
    data_root: str | Path,
    split_name: str,
    target_classes: list[str],
    all_classes: list[str],
    device: str = "cuda:0",
    max_crops_per_class: int = 30,
    min_crop_size: int = 32,
    num_templates: int = 1,
) -> dict[str, torch.Tensor]:
    """Generate Florence-2 captions with N task prompts, encode, and aggregate.

    For each crop: run N caption tasks → N captions → N BART embeddings → average.
    For each class: average all crop embeddings.

    Args:
        model_path: Path to local Florence-2 model directory.
        data_root: Dataset root with manifests/ and labels/.
        split_name: Data split name (base_train, novel_finetune).
        target_classes: Classes to generate descriptions for.
        all_classes: Full class list for index mapping.
        device: Torch device.
        max_crops_per_class: Max crops to caption per class.
        min_crop_size: Minimum crop side length in pixels.
        num_templates: Number of caption task prompts to use (1–3).

    Returns:
        Dict class_name → aggregated description embedding.
    """
    from fsod.florence2 import resolve_local_model_dir
    from fsod.modules.prototype import _load_labels
    from transformers import AutoModelForCausalLM, AutoProcessor

    if num_templates < 1 or num_templates > len(CAPTION_TASKS):
        raise ValueError(f"num_templates must be 1–{len(CAPTION_TASKS)}, got {num_templates}")

    tasks = CAPTION_TASKS[:num_templates]
    print(f"  Using {num_templates} caption task(s): {tasks}")

    resolved_path = str(resolve_local_model_dir(model_path))
    processor = AutoProcessor.from_pretrained(
        resolved_path, trust_remote_code=True, local_files_only=True,
    )
    fl_model = AutoModelForCausalLM.from_pretrained(
        resolved_path, dtype=torch.float32,
        trust_remote_code=True, local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    fl_model.eval()

    encoder = fl_model.language_model.get_encoder()

    data_root = Path(data_root)
    manifest_path = data_root / "manifests" / f"{split_name}.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    image_paths = [
        Path(line.strip())
        for line in manifest_path.read_text(encoding="utf-8").strip().splitlines()
    ]

    cls_idx_to_name = {i: name for i, name in enumerate(all_classes)}
    target_set = set(target_classes)

    # --- Phase 1: Collect crops per class ---
    class_crops: dict[str, list[Image.Image]] = {name: [] for name in target_classes}

    for img_path in image_paths:
        label_path = data_root / "labels" / split_name / (img_path.stem + ".txt")
        labels = _load_labels(label_path)
        if not labels:
            continue

        has_target = any(
            cls_idx_to_name.get(cls_id) in target_set for cls_id, *_ in labels
        )
        if not has_target:
            continue

        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        for cls_id, cx, cy, w, h in labels:
            cls_name = cls_idx_to_name.get(cls_id)
            if cls_name is None or cls_name not in target_set:
                continue
            if len(class_crops[cls_name]) >= max_crops_per_class:
                continue

            x1 = max(0, int((cx - w / 2) * orig_w))
            y1 = max(0, int((cy - h / 2) * orig_h))
            x2 = min(orig_w, int((cx + w / 2) * orig_w))
            y2 = min(orig_h, int((cy + h / 2) * orig_h))

            if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                continue

            class_crops[cls_name].append(img_pil.crop((x1, y1, x2, y2)))

    # --- Phase 2: Caption each crop with N tasks and encode ---
    class_embeddings: dict[str, list[torch.Tensor]] = {name: [] for name in target_classes}
    total_crops = sum(len(v) for v in class_crops.values())
    processed = 0

    for cls_name in target_classes:
        crops = class_crops[cls_name]

        if not crops:
            print(f"  WARNING: No crops for '{cls_name}', using class name embedding")
            tokens = processor.tokenizer(
                cls_name, return_tensors="pt", padding=False,
            ).to(device)
            enc_out = encoder(input_ids=tokens["input_ids"], return_dict=True)
            class_embeddings[cls_name].append(
                enc_out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            )
            continue

        for crop in crops:
            crop_embs: list[torch.Tensor] = []

            for task_prompt in tasks:
                # Generate caption
                inputs = processor(
                    text=task_prompt, images=crop, return_tensors="pt",
                ).to(device)
                generated_ids = fl_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=128,
                    num_beams=3,
                    do_sample=False,
                    use_cache=False,
                )
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=False,
                )[0]
                parsed = processor.post_process_generation(
                    generated_text, task=task_prompt, image_size=crop.size,
                )
                caption = parsed.get(task_prompt, cls_name)

                # Encode caption through text encoder
                tokens = processor.tokenizer(
                    caption, return_tensors="pt", padding=False,
                ).to(device)
                enc_out = encoder(input_ids=tokens["input_ids"], return_dict=True)
                emb = enc_out.last_hidden_state.mean(dim=1).squeeze(0)
                crop_embs.append(emb.cpu())

            # Average across templates for this crop
            crop_avg = torch.stack(crop_embs).mean(dim=0)
            class_embeddings[cls_name].append(crop_avg)
            processed += 1

        print(f"  {cls_name}: {len(crops)} crops × {num_templates} templates → {processed}/{total_crops} done")

    # --- Phase 3: Average per class ---
    result: dict[str, torch.Tensor] = {}
    emb_dim = None
    for cls_name in target_classes:
        embs = class_embeddings[cls_name]
        if embs:
            result[cls_name] = torch.stack(embs).mean(dim=0)
            emb_dim = result[cls_name].shape[0]
    for cls_name in target_classes:
        if cls_name not in result:
            result[cls_name] = torch.zeros(emb_dim or 768)

    del fl_model, encoder, processor
    torch.cuda.empty_cache()

    print(f"  Description embeddings: {len(result)} classes, dim={emb_dim}")
    return result


# ---------------------------------------------------------------------------
# CLI and training orchestration (derived from train_fsod.py)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO-FSOD with multi-template Florence-2 descriptions."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    parser.add_argument(
        "--stage", type=str, default="finetune",
        choices=["finetune"], help="Training stage (finetune only).",
    )
    parser.add_argument(
        "--base-weights", type=str, default="",
        help="Base pretrain checkpoint. Defaults to runs/voc_fsod_10shot/base_pretrain/weights/best.pt.",
    )
    parser.add_argument(
        "--model-arch", type=str, default="configs/yolo11s-fsod.yaml",
        help="Model architecture YAML.",
    )
    parser.add_argument(
        "--prototype", action="store_true",
        help="Initialize cosine head with visual prototypes + FiLM fusion.",
    )
    parser.add_argument(
        "--florence2", type=str, default="",
        help="Path to Florence-2 model directory.",
    )
    parser.add_argument(
        "--epochs", type=int, default=0,
        help="Override finetune epochs.",
    )
    parser.add_argument(
        "--num-templates", type=int, default=1,
        help="Number of Florence-2 caption tasks (1–3).",
    )
    parser.add_argument(
        "--run-suffix", type=str, default="",
        help="Append suffix to run directory name.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def get_all_classes(config: dict) -> list[str]:
    if "all_classes" in config:
        return config["all_classes"]
    if "coco_root" in config:
        from fsod.coco import COCO_CLASSES
        return list(COCO_CLASSES)
    from fsod.voc import VOC_CLASSES
    return list(VOC_CLASSES)


def get_yaml_prefix(config: dict) -> str:
    if "coco_root" in config:
        return "coco_fsod"
    return "voc_fsod"


def main() -> None:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))
    output_root = resolve_repo_path(config["output_root"])
    runs_dir = resolve_repo_path(config["runs_dir"])
    model_arch = resolve_repo_path(args.model_arch)
    device_str = f"cuda:{config['device']}" if str(config["device"]).isdigit() else config["device"]

    if not output_root.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {output_root}. Run prepare_voc_fewshot.py first."
        )

    novel_classes = config["novel_classes"]
    all_classes = get_all_classes(config)
    base_classes = [c for c in all_classes if c not in novel_classes]
    finetune_epochs = args.epochs if args.epochs > 0 else int(config["epochs"]["finetune"])

    base_weights = (
        resolve_repo_path(args.base_weights)
        if args.base_weights
        else runs_dir / "base_pretrain" / "weights" / "best.pt"
    )
    if not base_weights.exists():
        raise FileNotFoundError(
            f"Base pretrain weights not found: {base_weights}. Run base pretrain first."
        )

    # --- Build run name ---
    use_florence2 = bool(args.florence2)
    use_prototype = args.prototype

    if use_florence2 and use_prototype:
        run_name = "novel_finetune_cosine_fused"
    elif use_florence2:
        run_name = "novel_finetune_cosine_florence2"
    elif use_prototype:
        run_name = "novel_finetune_cosine_proto"
    else:
        run_name = "novel_finetune_cosine"

    if args.num_templates >= 1:
        run_name += f"_mt{args.num_templates}"
    if args.run_suffix:
        run_name += f"_{args.run_suffix}"
    if args.epochs > 0:
        run_name += f"_ep{args.epochs}"

    print(f"Run name: {run_name}")
    print(f"Num templates: {args.num_templates}")

    # --- Build model and load base weights ---
    print(f"Creating model from architecture: {model_arch}")
    model = YOLO(str(model_arch))
    print(f"Loading base pretrain weights: {base_weights}")
    model.load(str(base_weights))

    # --- Florence-2 multi-template pipeline ---
    _proto_init_data = None
    _florence_init_fn = None

    if use_prototype and not use_florence2:
        from fsod.modules.prototype import extract_prototypes, init_cosine_head_with_prototypes

        print("Extracting class prototypes from support set...")
        prototypes = extract_prototypes(
            base_weights=base_weights,
            data_root=output_root,
            novel_classes=novel_classes,
            all_classes=all_classes,
            imgsz=int(config["image_size"]),
            device=device_str,
        )
        _proto_init_data = (prototypes, novel_classes, all_classes)

    if use_florence2:
        from fsod.modules.adaptation import (
            extract_base_prototypes,
            extract_target_weights,
            train_modulation_network,
            init_cosine_head_modulated,
        )
        from fsod.modules.prototype import extract_prototypes

        print(f"Step 1/6: Generating multi-template descriptions for base classes (base_train)...")
        base_desc = generate_multitemplate_descriptions(
            model_path=args.florence2,
            data_root=output_root,
            split_name="base_train",
            target_classes=base_classes,
            all_classes=all_classes,
            device=device_str,
            num_templates=args.num_templates,
        )

        print(f"Step 2/6: Generating multi-template descriptions for novel classes (novel_finetune)...")
        novel_desc = generate_multitemplate_descriptions(
            model_path=args.florence2,
            data_root=output_root,
            split_name="novel_finetune",
            target_classes=novel_classes,
            all_classes=all_classes,
            device=device_str,
            num_templates=args.num_templates,
        )
        desc_embs = {**base_desc, **novel_desc}

        print("Step 3/6: Extracting base class visual prototypes...")
        base_protos = extract_base_prototypes(
            base_weights=base_weights,
            data_root=output_root,
            base_classes=base_classes,
            all_classes=all_classes,
            imgsz=int(config["image_size"]),
            device=device_str,
        )

        print("Step 4/6: Extracting novel class visual prototypes...")
        novel_protos = extract_prototypes(
            base_weights=base_weights,
            data_root=output_root,
            novel_classes=novel_classes,
            all_classes=all_classes,
            imgsz=int(config["image_size"]),
            device=device_str,
        )

        print("Step 5/6: Training scale-specific FiLM modulation networks on base classes...")
        target_wts = extract_target_weights(
            base_weights=base_weights,
            target_classes=base_classes,
            all_classes=all_classes,
        )
        fl_cfg = config.get("florence2", {})
        fusion_mode = str(fl_cfg.get("fusion_mode", "learnable")).strip().lower()
        if fusion_mode not in {"learnable", "fixed", "init_only"}:
            raise ValueError(f"Unsupported florence2.fusion_mode: {fusion_mode}")
        film = train_modulation_network(
            desc_embeddings=desc_embs,
            visual_prototypes=base_protos,
            target_weights=target_wts,
            base_classes=base_classes,
            hidden_dim=int(fl_cfg.get("film_hidden_dim", 128)),
            epochs=int(fl_cfg.get("film_epochs", 300)),
            weight_decay=float(fl_cfg.get("film_weight_decay", 0.05)),
            device=device_str,
        )

        blend_alpha = float(fl_cfg.get("alpha", 0.5))
        if fusion_mode != "learnable":
            run_name += f"_{fusion_mode}"
        print(
            f"Step 6/6: Will initialize VLM fusion in {fusion_mode} mode "
            f"with alpha={blend_alpha:.2f} after model rebuild..."
        )
        _florence_init_fn = lambda m: init_cosine_head_modulated(
            model=m,
            film=film,
            desc_embeddings=desc_embs,
            visual_prototypes=novel_protos,
            novel_classes=novel_classes,
            all_classes=all_classes,
            alpha=blend_alpha,
            fusion_mode=fusion_mode,
        )

    # --- Register callback ---
    def _on_pretrain_routine_end(trainer: Any) -> None:
        if _proto_init_data is not None:
            from fsod.modules.prototype import init_cosine_head_with_prototypes
            protos, n_cls, a_cls = _proto_init_data
            print("Injecting prototype weights into rebuilt model...")

            class _FakeYOLO:
                def __init__(self, det_model):
                    self.model = det_model

            init_cosine_head_with_prototypes(
                model=_FakeYOLO(trainer.model),
                prototypes=protos,
                novel_classes=n_cls,
                all_classes=a_cls,
            )
        if _florence_init_fn is not None:
            print("Injecting Florence-2 modulated weights into rebuilt model...")

            class _FakeYOLO2:
                def __init__(self, det_model):
                    self.model = det_model

            _florence_init_fn(_FakeYOLO2(trainer.model))

        if (_proto_init_data is not None or _florence_init_fn is not None) and getattr(trainer, "ema", None) is not None:
            model_sd = trainer.model.state_dict()
            for k, v in trainer.ema.ema.state_dict().items():
                if k in model_sd:
                    v.copy_(model_sd[k])
            print("Synced injected weights (incl. bool buffers) to EMA model")

    model.add_callback("on_pretrain_routine_end", _on_pretrain_routine_end)

    # --- Train ---
    model.train(
        data=str(output_root / f"{get_yaml_prefix(config)}_finetune.yaml"),
        epochs=finetune_epochs,
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]["finetune"]),
        workers=int(config["workers"]),
        device=config["device"],
        lr0=float(config["lr0"]["finetune"]),
        freeze=int(config.get("freeze", {}).get("backbone", 0)),
        patience=int(config.get("patience", {}).get("finetune", 30)),
        project=str(runs_dir),
        name=run_name,
        seed=int(config["seed"]),
        exist_ok=True,
    )

    best_path = runs_dir / run_name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Finetune stage finished but checkpoint not found: {best_path}")
    print(f"Finetune stage checkpoint: {best_path}")


if __name__ == "__main__":
    main()
