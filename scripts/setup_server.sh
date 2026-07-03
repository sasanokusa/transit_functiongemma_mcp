#!/usr/bin/env bash
set -euo pipefail

ROOT=${TRANSIT_ROOT:-/path/to/transit_functiongemma_mcp}
VENV=${TRANSIT_VENV:-/path/to/venvs/transit-functiongemma}
CACHE=${HF_HOME:-/path/to/huggingface}
XDG_CACHE=${XDG_CACHE_HOME:-/path/to/cache}
PIP_CACHE=${PIP_CACHE_DIR:-/path/to/pip-cache}
TMP=${TRANSIT_TMP:-/path/to/tmp}

if [[ "$(pwd)" != "$ROOT" ]]; then
  echo "Run this script from $ROOT" >&2
  exit 1
fi

mkdir -p "$VENV" "$CACHE" "$XDG_CACHE" "$PIP_CACHE" "$TMP" "$ROOT/outputs" "$ROOT/artifacts"
export HF_HOME="$CACHE"
export XDG_CACHE_HOME="$XDG_CACHE"
export PIP_CACHE_DIR="$PIP_CACHE"
export TMPDIR="$TMP"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
echo "Install the CUDA-compatible PyTorch wheel first if the default wheel is unsuitable."
python -m pip install -r requirements.txt

cat <<EOF
Environment created. Before each run:
  source $VENV/bin/activate
  export HF_HOME=$CACHE
  export XDG_CACHE_HOME=$XDG_CACHE
  export PIP_CACHE_DIR=$PIP_CACHE
EOF
