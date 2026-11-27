#!/usr/bin/env bash
set -e
BACKEND_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install --upgrade pip
python -m pip install -r "$BACKEND_DIR/requirements.txt"
python "$BACKEND_DIR/download_detector.py"
