"""Train YOLO-FSOD with Fused MLP+Prototype init (Approach B) and mosaic disabled.

Uses the simpler Fused approach instead of FiLM modulation:
  w_final = normalize(alpha * MLP(text) + (1-alpha) * visual_proto)

Pipeline:
  1. Extract Florence-2 text embeddings (class names → BART)
  2. Extract base class visual prototypes
  3. Train AdaptationMLP on base class (text, proto) pairs
  4. Extract novel class visual prototypes
  5. Initialize cosine head with fused MLP(text) + prototype weights
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_THIRD_PARTY_ULTRALYTICS = str(PROJECT_ROOT / "third_party" / "ultralytics")
if _THIRD_PARTY_ULTRALYTICS not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_ULTRALYTICS)

from ultralytics import YOLO

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO-FSOD with Fused MLP+Prototype init (no mosaic).")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    parser.add_argument(
        "--stage",
        type=str,
        default="finetune",
        choices=["all", "base", "finetune"],
        help="Training stage to run.",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default="",
        help="Base pretrain checkpoint for finetune. Defaults to runs/fsod_baseline/base_pretrain/weights/best.pt.",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default="configs/yolo11s-fsod.yaml",
        help="Model architecture YAML with FSODDetect head.",
    )
    parser.add_argument(
        "--prototype",
        action="store_true",
        help="Initialize cosine head with class prototypes from support set.",
    )
    parser.add_argument(
        "--florence2",
        type=str,
        default="",
        help="Path to Florence-2 model for adaptation init. Empty = disabled.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Override finetune epochs (0 = use config value).",
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
    if "yaml_prefix" in config:
        return config["yaml_prefix"]
    if "coco_root" in config:
        return "coco_fsod"
    return "voc_fsod"


def run_base_stage(config: dict, output_root: Path) -> Path:
    runs_dir = resolve_repo_path(config["runs_dir"])
    data_yaml = output_root / f"{get_yaml_prefix(config)}_base.yaml"

    model_source = config["model"]
    if str(model_source).endswith((".pt", ".yaml", ".yml")) and "/" in str(model_source):
        model_source = str(resolve_repo_path(model_source))

    model = YOLO(model_source)
    model.train(
        data=str(data_yaml),
        epochs=int(config["epochs"]["base"]),
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]["base"]),
        workers=int(config["workers"]),
        device=config["device"],
        lr0=float(config["lr0"]["base"]),
        project=str(runs_dir),
        name="base_pretrain",
        seed=int(config["seed"]),
        exist_ok=True,
    )

    best_path = runs_dir / "base_pretrain" / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Base stage finished but checkpoint not found: {best_path}")
    return best_path


def run_finetune_stage(
    config: dict, output_root: Path, base_weights: Path, model_arch: Path,
    use_prototype: bool = False, florence2_model: str = "",
    epochs_override: int = 0,
) -> Path:
    runs_dir = resolve_repo_path(config["runs_dir"])
    base_data_root = output_root
    if "base_data_root" in config:
        base_data_root = resolve_repo_path(config["base_data_root"])
    data_yaml = output_root / f"{get_yaml_prefix(config)}_finetune.yaml"
    novel_classes = config["novel_classes"]
    all_classes = get_all_classes(config)

    finetune_epochs = epochs_override if epochs_override > 0 else int(config["epochs"]["finetune"])

    # Run name — distinguish from FiLM version
    if florence2_model and use_prototype:
        run_name = "novel_finetune_cosine_mlpfuse_nomosaic"
    elif florence2_model:
        run_name = "novel_finetune_cosine_florence2_nomosaic"
    elif use_prototype:
        run_name = "novel_finetune_cosine_proto_nomosaic"
    else:
        run_name = "novel_finetune_cosine_nomosaic"

    if epochs_override > 0:
        run_name += f"_ep{epochs_override}"

    print(f"Creating model from architecture: {model_arch}")
    model = YOLO(str(model_arch))

    print(f"Loading base pretrain weights: {base_weights}")
    model.load(str(base_weights))

    _proto_init_data = None
    _florence_init_fn = None
    _alpha_log_enabled = False

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
            extract_florence2_text_embeddings,
            extract_base_prototypes,
            train_adaptation_mlp,
            init_cosine_head_fused,
        )
        from fsod.modules.prototype import extract_prototypes

        device_str = f"cuda:{config['device']}" if str(config["device"]).isdigit() else config["device"]
        base_classes = [c for c in all_classes if c not in novel_classes]

        print("Step 1/4: Extracting Florence-2 text embeddings for all classes...")
        all_names = base_classes + novel_classes
        text_embeddings = extract_florence2_text_embeddings(
            model_path=florence2_model,
            class_names=all_names,
            device=device_str,
        )

        print("Step 2/4: Extracting base class visual prototypes...")
        base_protos = extract_base_prototypes(
            base_weights=base_weights,
            data_root=base_data_root,
            base_classes=base_classes,
            all_classes=all_classes,
            imgsz=int(config["image_size"]),
            device=device_str,
        )

        print("Step 3/4: Training adaptation MLP on base classes...")
        fl_cfg = config.get("florence2", {})
        mlp = train_adaptation_mlp(
            text_embeddings=text_embeddings,
            visual_prototypes=base_protos,
            base_classes=base_classes,
            hidden_dim=int(fl_cfg.get("film_hidden_dim", 128)),
            epochs=int(fl_cfg.get("film_epochs", 300)),
            weight_decay=float(fl_cfg.get("film_weight_decay", 0.05)),
            device=device_str,
        )

        print("Step 4/4: Extracting novel class visual prototypes...")
        novel_protos = extract_prototypes(
            base_weights=base_weights,
            data_root=output_root,
            novel_classes=novel_classes,
            all_classes=all_classes,
            imgsz=int(config["image_size"]),
            device=device_str,
        )

        blend_alpha = float(fl_cfg.get("alpha", 0.5))
        print(f"Will initialize with fused MLP+prototype (alpha={blend_alpha:.2f}) after model rebuild...")
        _florence_init_fn = lambda m: init_cosine_head_fused(
            model=m,
            mlp=mlp,
            text_embeddings=text_embeddings,
            visual_prototypes=novel_protos,
            novel_classes=novel_classes,
            all_classes=all_classes,
            alpha=blend_alpha,
        )

    # --- Register callback ---
    def _on_pretrain_routine_end(trainer):
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
            print("Injecting Florence-2 fused MLP weights into rebuilt model...")
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
        mosaic=0.0,
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
        best_path = run_finetune_stage(
            config, output_root, base_weights, model_arch,
            use_prototype=args.prototype, florence2_model=args.florence2,
            epochs_override=args.epochs,
        )
        print(f"Finetune stage checkpoint: {best_path}")
        return

    # stage == "all"
    base_best = run_base_stage(config, output_root)
    finetune_best = run_finetune_stage(
        config, output_root, base_best, model_arch,
        use_prototype=args.prototype, florence2_model=args.florence2,
        epochs_override=args.epochs,
    )
    print(f"Base stage checkpoint: {base_best}")
    print(f"Finetune stage checkpoint: {finetune_best}")


if __name__ == "__main__":
    main()
