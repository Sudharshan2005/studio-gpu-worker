#!/usr/bin/env bash
# Pre-download all model weights onto the L40S box (into HF_HOME cache).
# Run ONCE after bootstrap; weights persist across runs so you don't re-pull.
# Only downloads what you'll use — comment out lines you don't need.
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "[download] HF cache: $HF_HOME"
pip install --quiet --upgrade "huggingface_hub[cli]"

dl(){ echo "[download] $1"; hf download "$1" --quiet || huggingface-cli download "$1"; }

dl "Wan-AI/Wan2.2-TI2V-5B-Diffusers"    # video  (~40GB)
# [session-trim: shot job does not use voice] dl "hexgrad/Kokoro-82M"                 # voice  (tiny)
dl "ACE-Step/ACE-Step-v1-3.5B"          # music  (~10GB)
# [session-trim: shot job does not use lipsync] dl "ByteDance/LatentSync-1.5"           # lipsync(~5GB)

echo "[download] done. df -h to confirm disk headroom before large runs."
