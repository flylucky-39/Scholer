"""Pre-compute CLIP text prototypes for a CD-FSOD dataset.

For each class, encode N prompt templates with the CLIP text encoder, average
the embeddings, L2-normalize, and save a tensor file used by ``CLIPVerifier``
at inference time.

Output schema (torch.save dict):
    {
        "proto":             FloatTensor [C, D]  L2-normalized text embeddings
        "class_names":       list[str]           ordered DIOR class names
        "class_name_to_idx": dict[str, int]
        "model_name":        str                 CLIP HF id
        "prompts_used":      list[list[str]]     [C][T] expanded prompts
        "embed_dim":         int
    }

Usage:
    python scripts/extract_vlm_text_proto.py \
        --dataset DIOR \
        --prompts prompts/dior.json \
        --model openai/clip-vit-base-patch32 \
        --out runs/vlm_assets/dior_clip_vitb32_text_proto.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fsod.cdfsod.datasets import get_dataset_spec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP text prototypes.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="CD-FSOD dataset name (e.g. DIOR, ArTaxOr).")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to prompt json (e.g. prompts/dior.json).")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                        help="HF CLIP model id or local model directory.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output .pt path.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for text encoder (cuda or cpu).")
    parser.add_argument("--cache-dir", type=str, default="",
                        help="Optional HuggingFace cache dir for offline loading.")
    parser.add_argument("--allow-online", action="store_true",
                        help="Allow remote download when model files are not cached locally.")
    return parser.parse_args()


def resolve_repo_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def main() -> None:
    args = parse_args()
    spec = get_dataset_spec(args.dataset)
    prompts_path = resolve_repo_path(args.prompts)
    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with prompts_path.open("r", encoding="utf-8") as f:
        prompt_cfg = json.load(f)

    class_labels = prompt_cfg["class_labels"]
    templates = prompt_cfg["templates"]

    # Validate every spec class has a readable label.
    missing = [c for c in spec.classes if c not in class_labels]
    if missing:
        raise KeyError(f"prompts/{args.dataset.lower()}.json missing labels for: {missing}")

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    local_files_only = not args.allow_online
    cache_dir = resolve_repo_path(args.cache_dir) if args.cache_dir else None
    print(
        f"Loading CLIP model: {args.model} (device={device}, "
        f"local_files_only={local_files_only})"
    )

    from transformers import CLIPModel, CLIPTokenizer
    try:
        model = CLIPModel.from_pretrained(
            args.model,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir else None,
        ).to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(
            args.model,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except OSError as exc:
        raise RuntimeError(
            "Failed to load CLIP model/tokenizer in offline mode. "
            "If server has no internet, pre-download the model on a networked machine "
            "and copy the full model directory to server, then pass that local directory "
            "to --model. Example: --model /root/models/clip-vit-base-patch32."
        ) from exc

    proto_list = []
    prompts_used = []
    for cls_name in spec.classes:
        label = class_labels[cls_name]
        cls_prompts = [t.format(label=label) for t in templates]
        prompts_used.append(cls_prompts)

        tok = tokenizer(cls_prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_text_features(**tok)  # [T, D]
        feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize each prompt
        avg = feat.mean(dim=0)
        avg = avg / avg.norm(dim=-1, keepdim=True)     # re-normalize after averaging
        proto_list.append(avg)

    proto = torch.stack(proto_list, dim=0).cpu()  # [C, D]
    embed_dim = proto.shape[1]

    payload = {
        "proto": proto,
        "class_names": list(spec.classes),
        "class_name_to_idx": {n: i for i, n in enumerate(spec.classes)},
        "model_name": args.model,
        "prompts_used": prompts_used,
        "embed_dim": embed_dim,
    }
    torch.save(payload, out_path)

    print(f"Saved text prototypes:")
    print(f"  path:        {out_path}")
    print(f"  shape:       {tuple(proto.shape)}  (C={len(spec.classes)}, D={embed_dim})")
    print(f"  classes:     {list(spec.classes)}")
    print(f"  templates:   {len(templates)} per class")


if __name__ == "__main__":
    main()
