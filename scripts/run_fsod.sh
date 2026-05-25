#!/bin/bash
# Wrapper to set up Python path and run FSOD training with local ultralytics
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/third_party/ultralytics:${PROJECT_ROOT}:$PYTHONPATH"

exec python3 "$@"
