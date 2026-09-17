"""
STUDIO GPU job runner — REAL model loading on the L40S.

Replaces the earlier stub. Each `kind` loads its model lazily (a worker doing
only 'voice' never loads the 40GB video model) and caches it for the process.

  shot     -> Wan 2.2 TI2V-5B (diffusers)  + optional character LoRA
  voice    -> Kokoro-82M
  music    -> ACE-Step 3.5B
  lipsync  -> LatentSync 1.5 (subprocess)
  assemble -> FFmpeg over the Scene JSON

⚠️ Model calls cannot be executed off-GPU. This code uses the real model IDs and
the documented library APIs, but MUST be smoke-tested on the L40S — treat the
first run of each kind as an integration test. Points needing on-box confirmation
are marked  # VERIFY-ON-BOX.
"""
import json
import os
import subprocess
from pathlib import Path

import models

# ---- lazy singletons, so we only pay VRAM for the kinds this worker runs ----
_WAN = None
_KOKORO = None
_ACESTEP = None


def _cuda_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device — this must run on the L40S box")
    return torch.cuda.get_device_name(0)


# ---------------------------------------------------------------- VIDEO (Wan)
def _load_wan():
    global _WAN
    if _WAN is not None:
        return _WAN
    import torch
    from diffusers import WanPipeline, AutoencoderKLWan
    print(f"[run] loading Wan {models.WAN_MODEL_ID} on {_cuda_check()}")
    vae = AutoencoderKLWan.from_pretrained(
        models.WAN_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        models.WAN_MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()   # headroom on a single 48GB card
    _WAN = pipe
    return pipe


def _gen_shot(payload: dict, out: Path) -> Path:
    from diffusers.utils import export_to_video
    pipe = _load_wan()

    # Character identity: load the per-character LoRA if the Scene JSON names one.
    lora = payload.get("lora")
    if lora:
        lora_path = Path(os.environ.get("LORA_DIR", "./loras")) / f"{lora}.safetensors"
        if lora_path.exists():
            pipe.load_lora_weights(str(lora_path))          # VERIFY-ON-BOX
            pipe.fuse_lora(lora_scale=payload.get("lora_scale", 0.7))
            print(f"[run] fused LoRA {lora} @ {payload.get('lora_scale',0.7)}")
        else:
            print(f"[run] WARNING: LoRA {lora} not found at {lora_path}")

    h, w = payload.get("height", models.WAN_DEFAULT_HW[0]), \
           payload.get("width", models.WAN_DEFAULT_HW[1])
    seconds = payload.get("duration", 5)
    num_frames = int(seconds * models.WAN_FPS) // 4 * 4 + 1   # Wan wants 4k+1 frames

    result = pipe(
        prompt=payload["prompt"],
        negative_prompt=payload.get("negative_prompt",
                                    "blurry, distorted, extra limbs, deformed"),
        height=h, width=w,
        num_frames=num_frames,
        guidance_scale=payload.get("guidance_scale", 5.0),
        num_inference_steps=payload.get("steps", 40),
    )
    frames = result.frames[0]                                # VERIFY-ON-BOX
    out = out.with_suffix(".mp4")
    export_to_video(frames, str(out), fps=models.WAN_FPS)
    if lora:
        pipe.unfuse_lora()
    return out


# ---------------------------------------------------------------- VOICE (Kokoro)
def _load_kokoro(lang="a"):
    global _KOKORO
    if _KOKORO is None:
        from kokoro import KPipeline
        print(f"[run] loading Kokoro {models.KOKORO_REPO}")
        _KOKORO = KPipeline(lang_code=lang)                  # 'a'=en, 'h'=hi ...
    return _KOKORO


def _gen_voice(payload: dict, out: Path) -> Path:
    import soundfile as sf
    import numpy as np
    kp = _load_kokoro(payload.get("lang_code", "a"))
    voice = payload.get("voice", models.KOKORO_VOICES["en_f"])
    chunks = []
    for _, _, audio in kp(payload["text"], voice=voice):     # VERIFY-ON-BOX
        chunks.append(audio)
    out = out.with_suffix(".wav")
    sf.write(str(out), np.concatenate(chunks), 24000)
    return out


# ---------------------------------------------------------------- MUSIC (ACE-Step)
def _gen_music(payload: dict, out: Path) -> Path:
    # ACE-Step ships its own pipeline (installed from source in bootstrap).
    from acestep.pipeline_ace_step import ACEStepPipeline    # VERIFY-ON-BOX
    global _ACESTEP
    if _ACESTEP is None:
        _ACESTEP = ACEStepPipeline(checkpoint_dir=os.environ.get("ACESTEP_CKPT"))
    out = out.with_suffix(".wav")
    _ACESTEP(
        prompt=payload.get("tags", "gentle childrens music, cheerful, acoustic"),
        lyrics=payload.get("lyrics", ""),
        audio_duration=payload.get("duration", 30),
        save_path=str(out),
    )
    return out


# ---------------------------------------------------------------- LIPSYNC (LatentSync)
def _gen_lipsync(payload: dict, out: Path) -> Path:
    # LatentSync is a repo, not a pip pkg -> call its inference script.
    ls_dir = Path(os.environ.get("LATENTSYNC_DIR", "./LatentSync"))
    out = out.with_suffix(".mp4")
    subprocess.run([                                          # VERIFY-ON-BOX
        "python", str(ls_dir / "inference.py"),
        "--video_path", payload["video"],
        "--audio_path", payload["audio"],
        "--video_out_path", str(out),
    ], check=True, cwd=str(ls_dir))
    return out


# ---------------------------------------------------------------- ASSEMBLE (FFmpeg)
def _resolve_in_outputs(name_or_id, channel: str, kind: str, out_dir: Path) -> Path:
    """A shot/music entry may be an explicit path, or a job-id we turn into the
    worker's own output filename (job{id}_{channel}_{kind}.mp4/.wav)."""
    s = str(name_or_id)
    if "/" in s or s.endswith((".mp4", ".wav")):        # explicit path/filename
        p = Path(s)
        return p if p.is_absolute() else (out_dir / p.name)
    ext = "wav" if kind in ("voice", "music") else "mp4"
    return out_dir / f"job{s}_{channel}_{kind}.{ext}"


def _assemble(payload: dict, out: Path) -> Path:
    """Concat shots into one video, optionally mixing a music track.

    Decoupled from the operator: pass `shot_ids` (job ids) + `channel` and the
    worker resolves the files from its OWN outputs dir — the operator never needs
    the box's paths. `shots` (explicit paths) still works. Music via `music_id`
    (a music job's id) or `music` (a path). Output is ONE combined mp4.
    """
    out_dir = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
    channel = payload.get("channel", "")

    entries = payload.get("shot_ids") or payload.get("shots")
    if not entries:
        raise ValueError("assemble needs 'shot_ids' or 'shots'")
    shots = [_resolve_in_outputs(e, channel, "shot", out_dir) for e in entries]
    for s in shots:
        if not s.exists():
            raise FileNotFoundError(f"assemble: shot not found: {s}")

    listfile = out.with_suffix(".txt")
    listfile.write_text("".join(f"file '{s.resolve()}'\n" for s in shots))
    out = out.with_suffix(".mp4")

    music = None
    if payload.get("music_id"):
        music = _resolve_in_outputs(payload["music_id"], channel, "music", out_dir)
    elif payload.get("music"):
        music = _resolve_in_outputs(payload["music"], channel, "music", out_dir)

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile)]
    if music and Path(music).exists():
        cmd += ["-i", str(music), "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-c:v", "libx264", "-an"]
    cmd += ["-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    listfile.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------- dispatch
_DISPATCH = {
    "shot": _gen_shot, "voice": _gen_voice, "music": _gen_music,
    "lipsync": _gen_lipsync, "assemble": _assemble,
}


def run_job(channel: str, kind: str, payload: dict, out_dir: Path, job_id: int) -> Path:
    fn = _DISPATCH.get(kind)
    if not fn:
        raise ValueError(f"unknown kind: {kind}")
    base = out_dir / f"job{job_id}_{channel}_{kind}"
    return fn(payload, base)
