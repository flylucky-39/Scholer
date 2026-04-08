"""Quick smoke test for prototype extraction."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import fsod.modules
from ultralytics import YOLO
from fsod.modules.prototype import _extract_cv3_features

out = []
try:
    model = YOLO("configs/yolo11s-fsod.yaml")
    model.model.eval()
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        feats = _extract_cv3_features(model, dummy)
    for i, f in enumerate(feats):
        out.append(f"Scale {i}: {tuple(f.shape)}")
    out.append("OK")
except Exception as e:
    import traceback
    out.append(f"ERROR: {e}")
    out.append(traceback.format_exc())

result = "\n".join(out)
(ROOT / "test_proto_result.txt").write_text(result, encoding="utf-8")
print(result)
