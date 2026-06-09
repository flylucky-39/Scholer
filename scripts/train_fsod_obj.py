"""Train YOLO-FSOD with Objectness branch (FSODDetectWithObjectness).

Adds class-agnostic objectness to decouple detection from classification,
enabling better few-shot performance.

Usage:
  # Base pretrain (same as standard FSOD)
  python scripts/train_fsod_obj.py --config configs/baseline_voc_10shot.yaml --stage base

  # Finetune with objectness head
  python scripts/train_fsod_obj.py --config configs/baseline_voc_10shot.yaml --stage finetune

  # Finetune with prototype init + objectness
  python scripts/train_fsod_obj.py --config configs/baseline_voc_10shot.yaml --stage finetune --prototype
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from ultralytics import YOLO

import fsod.modules  # noqa: F401
from fsod.modules.objectness_head import FSODDetectWithObjectness, add_objectness_callback

# Reuse helper functions from train_fsod
from scripts.train_fsod import (
    PROJECT_ROOT,
    load_config,
    resolve_repo_path,
    get_all_classes,
    get_yaml_prefix,
    run_base_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO-FSOD with Objectness branch.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    parser.add_argument(
        "--stage", type=str, default="finetune", choices=["all", "base", "finetune"],
        help="Training stage to run.",
    )
    parser.add_argument(
        "--base-weights", type=str, default="",
        help="Base pretrain checkpoint for finetune.",
    )
    parser.add_argument(
        "--model-arch", type=str, default="configs/yolo11s-fsod-obj.yaml",
        help="Model architecture YAML with FSODDetectWithObjectness head.",
    )
    parser.add_argument(
        "--prototype", action="store_true",
        help="Initialize cosine head with class prototypes from support set.",
    )
    parser.add_argument(
        "--florence2", type=str, default="",
        help="Path to Florence-2 model for adaptation init (disabled if empty).",
    )
    parser.add_argument(
        "--epochs", type=int, default=0,
        help="Override finetune epochs (0 = use config value).",
    )
    return parser.parse_args()


def run_finetune_stage_with_objectness(
    config: dict, output_root: Path, base_weights: Path, model_arch: Path,
    use_prototype: bool = False, florence2_model: str = "",
    epochs_override: int = 0,
) -> Path:
    """Finetune with FSODDetectWithObjectness (cosine classifier + objectness)."""
    runs_dir = resolve_repo_path(config["runs_dir"])
    base_data_root = output_root
    if "base_data_root" in config:
        base_data_root = resolve_repo_path(config["base_data_root"])
    data_yaml = output_root / f"{get_yaml_prefix(config)}_finetune.yaml"
    novel_classes = config["novel_classes"]
    all_classes = get_all_classes(config)

    finetune_epochs = epochs_override if epochs_override > 0 else int(config["epochs"]["finetune"])

    # Run name prefix
    run_name = "obj_finetune"
    if florence2_model:
        run_name += "_florence2"
    elif use_prototype:
        run_name += "_proto"

    # Handle fusion_mode from florence2 config
    _alpha_log_enabled = False
    if florence2_model:
        fl_cfg = config.get("florence2", {})
        fusion_mode = str(fl_cfg.get("fusion_mode", "learnable")).strip().lower()
        if fusion_mode not in {"learnable", "fixed", "init_only"}:
            raise ValueError(f"Unsupported florence2.fusion_mode: {fusion_mode}")
        if fusion_mode != "learnable":
            run_name += f"_{fusion_mode}"

    if epochs_override > 0:
        run_name += f"_ep{epochs_override}"

    print(f"Creating model from architecture: {model_arch}")
    model = YOLO(str(model_arch))

    print(f"Loading base pretrain weights: {base_weights}")
    model.load(str(base_weights))

    # --- Shared callback state (same as train_fsod.py) ---
    _proto_init_data = None
    _florence_init_fn = None

    if use_prototype and not florence2_model:
        from fsod.modules.prototype import extract_prototypes, init_cosine_head_with_prototypes

        print("Extracting class prototypes from support set...")
        prototypes = extract_prototypes(
            base_weights=base_weights,
            data_root=output_root,
            novel_classes=novel_classes,
            all_classes=all_classes,
            imgsz=int(config["image_size"]),
            device=f"cuda:{config['device']}" if str(config["device"]).isdigit() else config["device"],
        )
        _proto_init_data = (prototypes, novel_classes, all_classes)

    if florence2_model:
        from fsod.modules.adaptation import (
            generate_and_encode_descriptions,
            extract_base_prototypes,
            extract_target_weights,
            train_modulation_network,
            init_cosine_head_modulated,
        )
        from fsod.modules.prototype import extract_prototypes

        device_str = f"cuda:{config['device']}" if str(config["device"]).isdigit() else config["device"]
        base_classes = [c for c in all_classes if c not in novel_classes]

        print("Step 1/6: Generating descriptions for base classes (base_train)...")
        base_desc = generate_and_encode_descriptions(
            model_path=florence2_model,
            data_root=base_data_root,
            split_name="base_train",
            target_classes=base_classes,
            all_classes=all_classes,
            device=device_str,
        )

        print("Step 2/6: Generating descriptions for novel classes (novel_finetune)...")
        novel_desc = generate_and_encode_descriptions(
            model_path=florence2_model,
            data_root=output_root,
            split_name="novel_finetune",
            target_classes=novel_classes,
            all_classes=all_classes,
            device=device_str,
        )
        desc_embs = {**base_desc, **novel_desc}

        print("Step 3/6: Extracting base class visual prototypes...")
        base_protos = extract_base_prototypes(
            base_weights=base_weights,
            data_root=base_data_root,
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
        _alpha_log_enabled = bool(fl_cfg.get("alpha_logging", False))

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

    # --- Callback: inject prototype/florence weights after trainer rebuilds model ---
    def _on_pretrain_routine_end(trainer):
        """Inject custom CosineConv2d weights and set objectness criterion."""
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

        # Sync to EMA
        if (_proto_init_data is not None or _florence_init_fn is not None) and getattr(trainer, "ema", None) is not None:
            model_sd = trainer.model.state_dict()
            for k, v in trainer.ema.ema.state_dict().items():
                if k in model_sd:
                    v.copy_(model_sd[k])
            print("Synced injected weights (incl. bool buffers) to EMA model")

        # Set objectness criterion if the head supports it
        if isinstance(trainer.model.model[-1], FSODDetectWithObjectness):
            from fsod.modules.objectness_head import FSODObjectnessLoss
            trainer.model.criterion = FSODObjectnessLoss(trainer.model)
            trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss", "obj_loss")
            print("Using FSODObjectnessLoss with class-agnostic objectness")

    def _on_train_epoch_end(trainer):
        if not _alpha_log_enabled:
            return
        detect = trainer.model.model[-1]
        alpha_parts = []
        for scale_idx in range(detect.nl):
            cosine_layer = detect.cv3[scale_idx][-1]
            if not hasattr(cosine_layer, "get_blend_alpha"):
                continue
            alpha = cosine_layer.get_blend_alpha()
            active = getattr(cosine_layer, "has_active_prior", lambda: False)()
            state = "active" if active else "inactive"
            alpha_parts.append(f"P{scale_idx + 3}={alpha:.4f}({state})")
        if alpha_parts:
            epoch_idx = getattr(trainer, "epoch", -1) + 1
            print(f"[alpha] epoch={epoch_idx}: " + ", ".join(alpha_parts))

    model.add_callback("on_pretrain_routine_end", _on_pretrain_routine_end)
    model.add_callback("on_train_epoch_end", _on_train_epoch_end)

    model.train(
        data=str(data_yaml),
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
    return best_path


def main() -> None:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))
    output_root = resolve_repo_path(config["output_root"])
    runs_dir = resolve_repo_path(config["runs_dir"])

    if not output_root.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {output_root}. Run scripts/prepare_voc_fewshot.py first."
        )

    model_arch = resolve_repo_path(args.model_arch)

    if args.stage == "base":
        best_path = run_base_stage(config, output_root)
        print(f"Base stage checkpoint: {best_path}")
        return

    if args.stage == "finetune":
        base_weights = (
            resolve_repo_path(args.base_weights)
            if args.base_weights
            else runs_dir / "base_pretrain" / "weights" / "best.pt"
        )
        if not base_weights.exists():
            raise FileNotFoundError(
                f"Base pretrain weights not found: {base_weights}. Run --stage base first."
            )
        best_path = run_finetune_stage_with_objectness(
            config, output_root, base_weights, model_arch,
            use_prototype=args.prototype, florence2_model=args.florence2,
            epochs_override=args.epochs,
        )
        print(f"Finetune stage checkpoint: {best_path}")
        return

    # stage == "all"
    base_best = run_base_stage(config, output_root)
    finetune_best = run_finetune_stage_with_objectness(
        config, output_root, base_best, model_arch,
        use_prototype=args.prototype, florence2_model=args.florence2,
        epochs_override=args.epochs,
    )
    print(f"Base stage checkpoint: {base_best}")
    print(f"Finetune stage checkpoint: {finetune_best}")


if __name__ == "__main__":
    main()
