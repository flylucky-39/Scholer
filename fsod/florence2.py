from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


TASK_TOKENS = {
    "od": "<OD>",
    "region_proposal": "<REGION_PROPOSAL>",
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "caption_to_phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
}


@dataclass
class Florence2Prediction:
    image_path: str
    task: str
    task_token: str
    text_input: str
    raw_text: str
    parsed: dict[str, Any]
    detections: list[dict[str, Any]]


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtype_map[dtype_name]


def normalize_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def resolve_local_model_dir(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Florence-2 model path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Florence-2 model path is not a directory: {path}")

    if (path / "config.json").exists():
        return path

    snapshots_dir = path / "snapshots"
    if snapshots_dir.is_dir():
        snapshot_candidates = sorted(
            [candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir() and (candidate / "config.json").exists()]
        )
        if snapshot_candidates:
            return snapshot_candidates[-1]

    raise FileNotFoundError(
        "Florence-2 model directory does not contain config.json and no usable snapshots/* subdirectory was found: "
        f"{path}"
    )


def extract_detections(parsed_answer: dict[str, Any], task_token: str, width: int, height: int) -> list[dict[str, Any]]:
    task_result = parsed_answer.get(task_token, {})
    bboxes = task_result.get("bboxes", [])
    labels = task_result.get("labels", [])
    detections = []

    for index, bbox in enumerate(bboxes):
        label = labels[index] if index < len(labels) else ""
        x1, y1, x2, y2 = bbox
        detections.append(
            {
                "index": index,
                "label": label,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_norm": normalize_bbox([float(x1), float(y1), float(x2), float(y2)], width, height),
                "width": float(max(0.0, x2 - x1)),
                "height": float(max(0.0, y2 - y1)),
            }
        )

    return detections


class Florence2Runner:
    def __init__(self, model_path: str | Path, device: str = "cuda:0", dtype_name: str = "bf16") -> None:
        resolved_model_dir = resolve_local_model_dir(model_path)
        self.model_path = str(resolved_model_dir)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = resolve_torch_dtype(dtype_name)

        if self.device == "cpu" and self.dtype != torch.float32:
            self.dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    def run(self, image_path: str | Path, task: str, text_input: str = "", max_new_tokens: int = 256, num_beams: int = 3) -> Florence2Prediction:
        if task not in TASK_TOKENS:
            raise ValueError(f"Unsupported Florence-2 task: {task}")

        task_token = TASK_TOKENS[task]
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        prompt = task_token if not text_input else f"{task_token}{text_input}"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            use_cache=False,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_token,
            image_size=(image.width, image.height),
        )

        detections = extract_detections(parsed_answer, task_token, image.width, image.height)
        return Florence2Prediction(
            image_path=str(image_path.resolve()),
            task=task,
            task_token=task_token,
            text_input=text_input,
            raw_text=generated_text,
            parsed=parsed_answer,
            detections=detections,
        )