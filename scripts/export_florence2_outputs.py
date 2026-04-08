from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fsod.florence2 import Florence2Runner, filter_detections_by_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch export Florence-2 outputs for FSOD experiments.")
    parser.add_argument("--model-path", type=str, required=True, help="Local Florence-2 model directory.")
    parser.add_argument("--config", type=str, default="configs/baseline_voc_10shot.yaml", help="Baseline config yaml.")
    parser.add_argument("--input-manifest", type=str, default="", help="Text file containing image paths, one per line.")
    parser.add_argument("--input-dir", type=str, default="", help="Directory of images to process if manifest is not given.")
    parser.add_argument("--output-dir", type=str, default="./data/florence2_outputs", help="Output directory for exported JSON files.")
    parser.add_argument(
        "--task",
        type=str,
        default="od",
        choices=["od", "region_proposal", "caption_to_phrase_grounding", "caption", "detailed_caption", "more_detailed_caption"],
        help="Florence-2 task to run. Default: od (object detection with post-filter).",
    )
    parser.add_argument("--class-names", type=str, default="", help="Comma-separated class names to filter OD results.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device, e.g. cuda:0 or cpu.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Model dtype.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
    parser.add_argument("--num-beams", type=int, default=3, help="Beam size.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on processed images.")
    return parser.parse_args()


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_image_paths(manifest_path: Path | None, input_dir: Path | None) -> list[Path]:
    if manifest_path is not None:
        return [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if input_dir is not None:
        return sorted([path for path in input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    raise ValueError("Either --input-manifest or --input-dir must be provided.")


def write_json(file_path: Path, content: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))

    manifest_path = resolve_repo_path(args.input_manifest) if args.input_manifest else None
    input_dir = resolve_repo_path(args.input_dir) if args.input_dir else None
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_image_paths(manifest_path, input_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    if not class_names:
        class_names = list(config.get("novel_classes", []))

    runner = Florence2Runner(
        model_path=resolve_repo_path(args.model_path),
        device=args.device,
        dtype_name=args.dtype,
    )

    summary = {
        "model_path": str(resolve_repo_path(args.model_path)),
        "task": args.task,
        "class_names": class_names,
        "images": len(image_paths),
        "results": [],
    }

    for image_path in tqdm(image_paths, desc="Exporting Florence-2 outputs"):
        image_result = {
            "image_path": str(image_path.resolve()),
            "predictions": [],
        }

        if args.task in ("od", "region_proposal"):
            # Run once per image, then filter by class names
            prediction = runner.run(
                image_path=image_path,
                task=args.task,
                text_input="",
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            all_detections = prediction.detections
            if class_names and args.task == "od":
                matched = filter_detections_by_classes(all_detections, class_names)
            else:
                matched = all_detections

            image_result["predictions"].append(
                {
                    "query": "",
                    "task": prediction.task,
                    "task_token": prediction.task_token,
                    "raw_text": prediction.raw_text,
                    "all_detections": all_detections,
                    "filtered_detections": matched,
                    "num_all": len(all_detections),
                    "num_filtered": len(matched),
                }
            )
        elif args.task == "caption_to_phrase_grounding":
            # Build a single sentence prompt containing all class names
            prompt_text = ", ".join(class_names) if class_names else ""
            prediction = runner.run(
                image_path=image_path,
                task=args.task,
                text_input=prompt_text,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            image_result["predictions"].append(
                {
                    "query": prompt_text,
                    "task": prediction.task,
                    "task_token": prediction.task_token,
                    "raw_text": prediction.raw_text,
                    "detections": prediction.detections,
                }
            )
        else:
            # Caption tasks
            prediction = runner.run(
                image_path=image_path,
                task=args.task,
                text_input="",
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            image_result["predictions"].append(
                {
                    "query": "",
                    "task": prediction.task,
                    "task_token": prediction.task_token,
                    "raw_text": prediction.raw_text,
                    "detections": prediction.detections,
                }
            )

        result_path = output_dir / "results" / f"{image_path.stem}.json"
        write_json(result_path, image_result)
        summary["results"].append(
            {
                "image_path": image_result["image_path"],
                "result_path": str(result_path.resolve()),
                "num_prediction_groups": len(image_result["predictions"]),
            }
        )

    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "images": len(image_paths), "task": args.task}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()