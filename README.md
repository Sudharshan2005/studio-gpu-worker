# STUDIO — GPU Worker (`github/`)

Runs on the rented L40S. Pulls jobs from the operator, loads the models, generates,
writes to `outputs/` for manual download. Ephemeral: wipe the whole clone when done.

## Models it loads (all Apache-2.0, commercial-safe)
| Job kind | Model | HF repo | ~VRAM |
|---|---|---|---|
| `shot` | Wan 2.2 TI2V-5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | ~40GB |
| `voice` | Kokoro-82M | `hexgrad/Kokoro-82M` | ~1GB |
| `music` | ACE-Step 3.5B | `ACE-Step/ACE-Step-v1-3.5B` | ~10GB |
| `lipsync` | LatentSync 1.5 | `ByteDance/LatentSync-1.5` | ~5GB |
| `assemble` | FFmpeg | — | — |

Full config in `models.py`. The 5B Wan variant is chosen because it fits ONE L40S
(48GB) and does both text→video and image→video. The 14B MoE needs bigger GPUs.

## Run
```bash
git clone <this repo> && cd github
./bootstrap.sh          # sources .env, installs stack, downloads weights, runs worker
# ... jobs process, outputs land in ./outputs/ ...
scp -r user@l40s:$(pwd)/outputs ./local   # download from your Mac
./teardown.sh "$(pwd)"                     # wipe clone + venv + docker, clear env
# then TERMINATE the instance in the provider console (stops billing)
```

## Secrets
`.env` is committed here (your setup) with placeholders — fill once:
`CONTROL_PLANE_URL`, `WORKER_TOKEN` (must match `operator/.env`). Generate with
`openssl rand -hex 32`.

## ⚠️ Model code status
`run_job.py` uses the real model IDs and documented library APIs, but GPU model
calls could not be executed off-box. Treat the first run of each `kind` as an
integration test; points needing confirmation are marked `# VERIFY-ON-BOX`.
Language note: Kokoro is weak on te/ta/kn — use Chatterbox for those (see models.py).

## Shared box: vai-dev2 (8× L40S, 46GB, driver 580 / CUDA 13.0)
This is NOT a rented instance you own — other tenants run vLLM/NIM/VST on other
GPUs. So:
- **Pin one free GPU by UUID** — `gpu_pick.py` does this automatically (bootstrap
  calls it). GPU 5 is broken, which shifts CUDA indices, so index pinning is
  unsafe; UUID pinning avoids it. Idle card default:
  `GPU-8b870518-304a-92a4-b17e-c2633fca54e1`.
- **teardown removes ONLY your footprint** — your process (by pidfile) + your
  scratch. It does NOT prune docker globally or kill the box. Freeing your GPU =
  your process exiting.
- **There is no instance to terminate.** "Leave no trace" here means remove your
  own clone/outputs/process, never touch other tenants' work.
- torch: driver 580 runs cu128 or cu124 wheels; requirements.txt uses cu128.
