"""Quick test: verify FSODDetect model builds correctly."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fsod.modules  # noqa: F401

from ultralytics import YOLO

out_lines = []

try:
    model = YOLO("configs/yolo11-fsod.yaml")
    head = model.model.model[-1]
    out_lines.append(f"Head type: {type(head).__name__}")
    out_lines.append(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    for i, cv3 in enumerate(head.cv3):
        last = cv3[-1]
        out_lines.append(f"  Scale {i} cls layer: {type(last).__name__}")
        if hasattr(last, "weight"):
            out_lines.append(f"    weight shape: {tuple(last.weight.shape)}")
        if hasattr(last, "scale"):
            out_lines.append(f"    temperature: {last.scale.exp().item():.1f}")
    out_lines.append("OK")
except Exception as e:
    out_lines.append(f"ERROR: {e}")
    import traceback
    out_lines.append(traceback.format_exc())

result = "\n".join(out_lines)
(ROOT / "test_result.txt").write_text(result, encoding="utf-8")
print(result hasattr(last, "weight"):
            out_lines.append(f"    weight shape: {tuple(last.weight.shape)}")
        if hasattr(last, "scale"):
            out_lines.append(f"    temperature: {last.scale.exp().item():.1f}")
    out_lines.append("OK")
except Exception as e:
    out_lines.append(f"ERROR: {e}")
    import traceback
    out_lines.append(traceback.format_exc())

result = "\n".join(out_lines)
(ROOT / "test_result.txt").write_text(result, encoding="utf-8")
print(result)
