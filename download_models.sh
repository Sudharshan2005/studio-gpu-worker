#!/usr/bin/env bash
# Pre-download all model weights onto the L40S box (into HF_HOME cache).
# Run ONCE after bootstrap; weights persist across runs so you don't re-pull.
# Only downloads what you'll use — comment out lines you don't need.
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "[download] HF cache: $HF_HOME"
pip install --quiet --upgrade "huggingface_hub[cli]"

dl(){ echo "[download] $1"; hf download "$1" --quiet || huggingface-cli download "$1"; }
# CHECK DISK FIRST, not after — a mid-download eviction on a shared node is ugly.
df -h / | tail -1

dl "Wan-AI/Wan2.2-TI2V-5B-Diffusers"    # video   (~32GiB)
dl "hexgrad/Kokoro-82M"                 # voice   (tiny)
dl "ByteDance/LatentSync-1.5"           # lipsync (~5GB)

# ACE-Step (music, ~7.7GiB) does NOT read HF_HOME — its pipeline resolves
# checkpoint_dir=None to ~/.cache/ace-step/checkpoints and expects a FLAT layout
# (music_dcae_f8c8, music_vocoder, ace_step_transformer, umt5-base). Pull it there
# with --local-dir, and set ACESTEP_CKPT to that path (or leave it to self-download
# on the first music job). A plain `dl` into HF_HOME would be downloaded twice.
ACESTEP_CKPT="${ACESTEP_CKPT:-$HOME/.cache/ace-step/checkpoints}"
hf download "ACE-Step/ACE-Step-v1-3.5B" --local-dir "$ACESTEP_CKPT" --quiet \
  || huggingface-cli download "ACE-Step/ACE-Step-v1-3.5B" --local-dir "$ACESTEP_CKPT"

echo "[download] done."
