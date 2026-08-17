#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV_PATH="$REPOSITORY_ROOT/.venv-local"
UV_BIN=${UV_BIN:-$HOME/.local/bin/uv}

if [[ ! -x "$UV_BIN" ]]; then
  python3 -m pip install --user --no-cache-dir uv
fi

cd "$REPOSITORY_ROOT"
"$UV_BIN" python install 3.12
"$UV_BIN" venv --python 3.12 "$VENV_PATH"

# Install CPU torch first so resolving the editable project does not pull the
# much larger CUDA wheel set on a workstation without an NVIDIA device.
UV_NO_CACHE=1 "$UV_BIN" pip install \
  --python "$VENV_PATH/bin/python" \
  torch \
  --index-url https://download.pytorch.org/whl/cpu

UV_NO_CACHE=1 "$UV_BIN" pip install \
  --python "$VENV_PATH/bin/python" \
  -e '.[rfd3]' \
  pytest \
  ruff==0.8.3

DEBUG=false TYPE_CHECK=false NAN_CHECK=true \
  "$VENV_PATH/bin/python" - <<'PY'
import sys
import torch
import pydantic
import rfd3
import rfd3_mosaic

print("Local RFD3-Mosaic CPU environment ready")
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("pydantic:", pydantic.__version__)
PY

echo
echo "Activate with:"
echo "  source scripts/rfd3_mosaic/activate_local_dev.sh"
