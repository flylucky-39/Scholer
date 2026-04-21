"""Train YOLO-FSOD on a CD-FSOD-Bench target domain with VLM-guided
Domain-Adaptive Fusion (DAF) and Cross-Domain Prototype Calibration (CDPC).

Pipeline
--------
    1. Load a base-pretrained checkpoint (e.g. COCO-base trained earlier).
    2. Estimate the domain gap g between source images and the target's
       few-shot support set (Domain-Gap Estimator, DGE).
    3. Compute α(g) via the DAF sigmoid mapping.
    4. Extract visual prototypes for the target novel classes.
    5. (Optional) Build text prototypes via Florence-2 / sentence templates.
    6. Apply CDPC to pull visual prototypes toward the text anchors.
    7. Fuse (1−α) · visual + α · textual and inject into CosineConv2d.
    8. Train YOLO11s on the K-shot target set, using the FSOD finetune recipe.

This script is read-only with respect to existing code; it composes the
existing modules ``fsod.modules.cosine_head``, ``fsod.modules.prototype``
and ``fsod.modules.adaptation`` together with the new ``fsod.cdfsod``
package.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401  ensure FSODDetect registered

from fsod.cdfsod import (  # noqa: E402
    alpha_from_gap,
    calibrate_prototypes,
    fuse_prototypes,
    get_dataset_spec,
)
from fsod.cdfsod.domain_gap import DomainGapEstimator  # noqa: E402


# ---------------------------------------------------------------------------
# CLI / config plumbing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CD-FSOD with DAF + CDPC.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--base-weights", type=str, required=True,
                   help="Source-domain pretrained checkpoint (e.g. COCO base best.pt).")
    p.add_argument("--model-arch", type=str, default="configs/yolo11s-fsod.yaml")
    p.add_argument("--no-text", action="store_true",
                   help="Disable VLM/text prototypes (visual-only baseline; α forced to 0).")
    p.add_argument("--no-dge", action="store_true",
                   help="Disable Domain-Gap Estimator; use config 'alpha_default' directly.")
    p.add_argument("--no-cdpc", action="store_true",
                   help="Disable Cross-Domain Prototype Calibration.")
    p.add_argument("--epochs", type=int, default=0,
                   help="Override finetune epochs (0 = use config).")
    return p.parse_args()


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Source / target image enumeration for DGE
# ---------------------------------------------------------------------------

def _read_manifest(manifest: Path) -> list[Path]:
    if not manifest.exists():
        return []
    return [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def gather_source_images(cfg: dict) -> list[Path]:
    """Source images for DGE — typically the source-domain train manifest.

    Resolution order:
      * cfg['source_image_manifest'] (explicit path to a *.txt)
      * cfg['source_image_dir'] (a directory; uses up to N=200 images)
    """
    if "source_image_manifest" in cfg:
        return _read_manifest(resolve_repo_path(cfg["source_image_manifest"]))
    if "source_image_dir" in cfg:
        d = resolve_repo_path(cfg["source_image_dir"])
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return [p for p in sorted(d.rglob("*")) if p.suffix.lower() in exts]
    raise KeyError("Config must define either 'source_image_manifest' or 'source_image_dir' for DGE.")


def gather_target_images(out_root: Path) -> list[Path]:
    return _read_manifest(out_root / "manifests" / "novel_finetune.txt")


# ---------------------------------------------------------------------------
# Text prototypes
# ---------------------------------------------------------------------------

def build_text_prototypes(
    classes: Iterable[str],
    *,
    feature_dim: int,
    florence2_model: str | None,
    device: str,
) -> dict[str, torch.Tensor] | None:
    """Construct per-class text prototypes in YOLO feature space.

    Two backends:

    1. **Florence-2 + simple linear projection** (preferred):
       Re-uses ``extract_florence2_text_embeddings`` if a model path is given.
       The 1024-dim Florence text vector is then projected to ``feature_dim``
       through a frozen random orthonormal matrix (deterministic, reproducible
       via fsod.cdfsod RNG seed). This is intentionally minimal — full FiLM
       training is left to ``train_fsod.py`` and is *not* required for CDPC.

    2. **None**: returns ``None`` (caller falls back to visual-only).
    """
    if not florence2_model:
        return None

    try:
        from fsod.modules.adaptation import extract_florence2_text_embeddings
    except ImportError:
        print("  [text] Florence-2 dependencies missing; skipping text prototypes.")
        return None

    class_list = list(classes)
    print(f"  [text] Encoding {len(class_list)} class names with Florence-2...")
    text_embs = extract_florence2_text_embeddings(florence2_model, class_list, device=device)

    # Build a frozen orthonormal projection 1024 → feature_dim.
    sample_dim = next(iter(text_embs.values())).shape[0]
    gen = torch.Generator().manual_seed(3407)
    proj = torch.empty(sample_dim, feature_dim).normal_(generator=gen)
    proj, _ = torch.linalg.qr(proj)  # 1024 × feature_dim, columns orthonormal
    proj = proj[:, :feature_dim]

    out: dict[str, torch.Tensor] = {}
    for name in class_list:
        v = text_embs[name].float() @ proj
        out[name] = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_repo_path(args.config))
    out_root = resolve_repo_path(cfg["output_root"])
    runs_dir = resolve_repo_path(cfg["runs_dir"])
    base_weights = resolve_repo_path(args.base_weights)
    model_arch = resolve_repo_path(args.model_arch)
    spec = get_dataset_spec(cfg["cdfsod_dataset"])

    if not out_root.exists():
        raise FileNotFoundError(
            f"Prepared CD-FSOD dataset not found: {out_root}. "
            "Run scripts/prepare_cdfsod.py first."
        )
    if not base_weights.exists():
        raise FileNotFoundError(f"Base weights not found: {base_weights}")

    device_str = (
        f"cuda:{cfg['device']}" if str(cfg["device"]).isdigit() else str(cfg["device"])
    )
    finetune_epochs = args.epochs if args.epochs > 0 else int(cfg["epochs"]["finetune"])

    # -------- (1) DGE -------------------------------------------------------
    if args.no_dge:
        g = float(cfg.get("alpha_default", 0.5))
        backend = "<DGE disabled — using alpha_default as gap stand-in>"
    else:
        try:
            src_paths = gather_source_images(cfg)
        except KeyError as e:
            raise SystemExit(
                f"DGE enabled but {e}. Add 'source_image_dir' or 'source_image_manifest' to the config, "
                "or pass --no-dge."
            )
        tgt_paths = gather_target_images(out_root)
        print(f"DGE: encoding {len(src_paths)} source / {len(tgt_paths)} target images")
        dge = DomainGapEstimator(device=device_str, max_images=int(cfg.get("dge_max_images", 200)))
        g = dge.estimate_gap(src_paths, tgt_paths)
        backend = dge.backend

    alpha_cfg = cfg.get("daf", {})
    k = float(alpha_cfg.get("k", 6.0))
    g0 = float(alpha_cfg.get("g0", 0.35))
    alpha = 0.0 if args.no_text else alpha_from_gap(g, k=k, g0=g0)
    print(f"DGE backend: {backend}")
    print(f"  domain_gap g = {g:.4f}  →  alpha(g) = {alpha:.4f}  (k={k}, g0={g0})")

    # -------- (2) Visual prototypes ----------------------------------------
    from fsod.modules.prototype import extract_prototypes

    print("Extracting visual prototypes from K-shot support...")
    visual_protos = extract_prototypes(
        base_weights=base_weights,
        data_root=out_root,
        novel_classes=list(spec.classes),
        all_classes=list(spec.classes),
        imgsz=int(cfg["image_size"]),
        device=device_str,
    )
    feature_dim = next(iter(visual_protos.values())).shape[0]

    # -------- (3) Text prototypes ------------------------------------------
    if args.no_text:
        text_protos = None
    else:
        florence2_path = cfg.get("florence2_model", "")
        text_protos = build_text_prototypes(
            spec.classes,
            feature_dim=feature_dim,
            florence2_model=florence2_path or None,
            device=device_str,
        )
        if text_protos is None:
            print("  [text] No text prototypes available; falling back to visual-only init.")
            alpha = 0.0

    # -------- (4) CDPC + fusion --------------------------------------------
    if text_protos is not None and not args.no_cdpc:
        cdpc_step = float(cfg.get("cdpc_step", 0.1))
        print(f"  CDPC: pulling visual prototypes toward text anchors (step={cdpc_step}, alpha={alpha:.3f})")
        visual_protos = calibrate_prototypes(visual_protos, text_protos, alpha=alpha, step=cdpc_step)

    if text_protos is not None:
        fused_protos = fuse_prototypes(visual_protos, text_protos, alpha=alpha)
    else:
        # Visual-only path: just L2-normalise.
        fused_protos = {n: F.normalize(v.unsqueeze(0), dim=1).squeeze(0) for n, v in visual_protos.items()}

    # -------- (5) Build / load YOLO model ----------------------------------
    print(f"Building model from {model_arch} (FSODDetect head, nc={spec.num_classes})")
    model = YOLO(str(model_arch))
    print(f"Loading base weights: {base_weights}")
    model.load(str(base_weights))

    # -------- (6) Inject prototypes via callback ---------------------------
    novel_classes = list(spec.classes)
    all_classes = list(spec.classes)

    def _inject(trainer):
        from fsod.modules.prototype import init_cosine_head_with_prototypes

        class _Wrap:
            def __init__(self, det_model):
                self.model = det_model

        init_cosine_head_with_prototypes(
            model=_Wrap(trainer.model),
            prototypes=fused_protos,
            novel_classes=novel_classes,
            all_classes=all_classes,
        )

        # Sync to EMA — bool buffers are not updated by ModelEMA.update().
        if getattr(trainer, "ema", None) is not None:
            sd = trainer.model.state_dict()
            for k_, v_ in trainer.ema.ema.state_dict().items():
                if k_ in sd:
                    v_.copy_(sd[k_])
            print("Synced fused prototypes (incl. bool buffers) to EMA.")

    model.add_callback("on_pretrain_routine_end", _inject)

    # -------- (7) Train ----------------------------------------------------
    run_name = _build_run_name(cfg, args, alpha)
    data_yaml = out_root / "cdfsod_bench_finetune.yaml"
    print(f"Training: {run_name}  →  {runs_dir / run_name}")

    model.train(
        data=str(data_yaml),
        epochs=finetune_epochs,
        imgsz=int(cfg["image_size"]),
        batch=int(cfg["batch_size"]["finetune"]),
        workers=int(cfg["workers"]),
        device=cfg["device"],
        lr0=float(cfg["lr0"]["finetune"]),
        freeze=int(cfg.get("freeze", {}).get("backbone", 0)),
        patience=int(cfg.get("patience", {}).get("finetune", 30)),
        project=str(runs_dir),
        name=run_name,
        seed=int(cfg["seed"]),
        exist_ok=True,
    )

    # -------- (8) Persist DGE/alpha info next to the run ------------------
    summary = {
        "domain_gap_g": g,
        "alpha": alpha,
        "k": k,
        "g0": g0,
        "dge_backend": backend,
        "no_text": bool(args.no_text),
        "no_dge": bool(args.no_dge),
        "no_cdpc": bool(args.no_cdpc),
        "dataset": spec.name,
        "shot": int(cfg["shot"]),
    }
    summary_path = runs_dir / run_name / "cdfsod_summary.yaml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


def _build_run_name(cfg: dict, args: argparse.Namespace, alpha: float) -> str:
    base = f"cdfsod_{cfg['cdfsod_dataset']}_{cfg['shot']}shot"
    parts: list[str] = []
    if args.no_text:
        parts.append("visual_only")
    else:
        parts.append(f"daf_a{alpha:.2f}")
        if args.no_cdpc:
            parts.append("noCDPC")
        if args.no_dge:
            parts.append("noDGE")
    return base + "_" + "_".join(parts) if parts else base


if __name__ == "__main__":
    main()
