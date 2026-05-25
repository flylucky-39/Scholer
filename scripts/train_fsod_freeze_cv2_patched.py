"""Sets up sys.path for vendored ultralytics, then runs train_fsod_freeze_cv2.py."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_THIRD_PARTY_ULTRALYTICS = str(_PROJECT_ROOT / "third_party" / "ultralytics")
if _THIRD_PARTY_ULTRALYTICS not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_ULTRALYTICS)

if __name__ == '__main__':
    script_path = str(_PROJECT_ROOT / 'scripts' / 'train_fsod_freeze_cv2.py')
    sys.argv[0] = script_path
    print(f'[patched] sys.argv={sys.argv}', flush=True)
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
    g = {
        '__name__': '__main__',
        '__file__': script_path,
        '__builtins__': __builtins__,
        '__loader__': None,
        '__package__': None,
        '__spec__': None,
    }
    exec(code, g)
