#!/usr/bin/env bash
# STUDIO GPU bootstrap — run ON the L40S box, inside this cloned repo.
# Sources ./.env for secrets (committed per your setup), installs the model
# stack, then runs the worker until the queue drains.
#
#   git clone <this repo> && cd github && ./bootstrap.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- secrets from committed .env (CONTROL_PLANE_URL, WORKER_TOKEN, HF_TOKEN) ---
[ -f .env ] && { set -a; . ./.env; set +a; echo "[bootstrap] sourced .env"; }
: "${CONTROL_PLANE_URL:?set CONTROL_PLANE_URL in .env}"
: "${WORKER_TOKEN:?set WORKER_TOKEN in .env}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# --- GPU sanity ---
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

# --- python env (persists in the clone; teardown wipes the whole clone) ---
python3 -m venv .venv && source .venv/bin/activate
pip install --quiet --upgrade pip

# torch matched to the box CUDA (edit --extra-index-url in requirements.txt if needed)
pip install --quiet -r requirements.txt

# models that aren't on PyPI (install once; skip if already present)
python3 -c "import acestep" 2>/dev/null || \
  pip install --quiet "git+https://github.com/ace-step/ACE-Step.git"
[ -d LatentSync ] || git clone --depth 1 https://github.com/bytedance/LatentSync.git
export LATENTSYNC_DIR="$HERE/LatentSync"
export LORA_DIR="${LORA_DIR:-$HERE/loras}"; mkdir -p "$LORA_DIR"

# weights (safe to re-run; huggingface caches)
[ "${SKIP_DOWNLOAD:-0}" = "1" ] || ./download_models.sh

export OUTPUT_DIR="$HERE/outputs"; mkdir -p "$OUTPUT_DIR"
echo "[bootstrap] starting worker -> $CONTROL_PLANE_URL"
python3 worker.py

echo "[bootstrap] queue drained. Outputs: $OUTPUT_DIR"
echo "[bootstrap] DOWNLOAD them (scp), then: ./teardown.sh $HERE"
