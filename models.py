"""
STUDIO GPU model registry — the EXACT models this worker loads on the L40S.

Every model here is open-weight and commercially usable (verified in
../MASTER_PLAN.md §10). VRAM notes are for a single L40S (48GB).

  VIDEO   Wan 2.2 TI2V-5B   -> Wan-AI/Wan2.2-TI2V-5B-Diffusers   (~40GB, fits)
  VOICE   Kokoro-82M        -> hexgrad/Kokoro-82M                (tiny)
  MUSIC   ACE-Step 3.5B     -> ACE-Step/ACE-Step-v1-3.5B         (~10GB)
  LIPSYNC LatentSync 1.5    -> ByteDance/LatentSync-1.5          (~5GB)

NOTE on model choice: the 5B TI2V variant is used (not the 14B A14B MoE) because
the 14B needs multi-GPU / heavy offload. The 5B does BOTH text->video and
image->video and fits one L40S. Swap to 14B only if you move to bigger GPUs.
"""

# --- Video: Wan 2.2 (Apache-2.0) ---
WAN_MODEL_ID   = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_DTYPE      = "bfloat16"
WAN_FPS        = 24
WAN_DEFAULT_HW = (704, 1280)   # H, W  (9:16 vertical -> (1280, 704))

# --- Voice: Kokoro-82M (Apache-2.0) ---
# NOTE: Kokoro's strong languages are en/es/fr/hi/it/pt/ja/zh. For te/ta/kn,
# results are weak -> see VOICE_FALLBACK. Do NOT ship low-quality kids audio.
KOKORO_REPO    = "hexgrad/Kokoro-82M"
KOKORO_VOICES  = {"en_f": "af_heart", "en_m": "am_michael", "hi_f": "hf_alpha"}
VOICE_FALLBACK = "Chatterbox (resemble-ai/chatterbox) for languages Kokoro lacks"

# --- Music: ACE-Step (Apache-2.0) ---
ACESTEP_REPO   = "ACE-Step/ACE-Step-v1-3.5B"

# --- Lip-sync: LatentSync (Apache-2.0) ---
LATENTSYNC_REPO = "ByteDance/LatentSync-1.5"

# Where weights are cached on the GPU box (persists across runs; set by bootstrap).
import os
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

ALL = {
    "video":   {"id": WAN_MODEL_ID,   "vram_gb": 40, "kind": "shot"},
    "voice":   {"id": KOKORO_REPO,    "vram_gb": 1,  "kind": "voice"},
    "music":   {"id": ACESTEP_REPO,   "vram_gb": 10, "kind": "music"},
    "lipsync": {"id": LATENTSYNC_REPO,"vram_gb": 5,  "kind": "lipsync"},
}
