#!/usr/bin/env bash
# STUDIO teardown — SHARED-BOX SAFE.
#
# vai-dev2 is a SHARED machine: other tenants run vLLM / NIM / VST on other GPUs.
# This script therefore removes ONLY your own footprint. It never prunes docker
# globally, never deletes containers it did not create, never kills the box.
#
# What it does: kill only your worker process, wipe your scratch/clone, clear
# your env secrets. Your GPU memory frees automatically when your process exits.
#
# Usage:
#   ./teardown.sh /path/to/clone [--pidfile worker.pid] [--container studio-worker-<you>]
set -euo pipefail

TARGET="${1:?pass the clone/scratch dir to remove}"; shift || true
PIDFILE=""; CONTAINER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --pidfile) PIDFILE="$2"; shift 2;;
    --container) CONTAINER="$2"; shift 2;;   # MUST be your own named container
    *) echo "unknown arg $1"; exit 1;;
  esac
done

# 1. Stop ONLY your worker process (by pidfile), never a blanket pkill.
if [ -n "${PIDFILE}" ] && [ -f "${PIDFILE}" ]; then
  PID="$(cat "${PIDFILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}" 2>/dev/null || true
    echo "[teardown] stopped worker pid ${PID} (GPU memory frees on exit)"
  fi
  rm -f "${PIDFILE}"
fi

# 2. Remove ONLY your own named container, if you ran one. No global prune.
if [ -n "${CONTAINER}" ] && command -v docker >/dev/null 2>&1; then
  docker rm -f "${CONTAINER}" 2>/dev/null && echo "[teardown] removed container ${CONTAINER}" || true
fi

# 3. Wipe your scratch/clone (venv, outputs, temp media). Refuse suspicious paths.
case "${TARGET}" in
  /|/home|/root|/tmp|"") echo "[teardown] refusing to remove '${TARGET}'"; exit 1;;
esac
if [ -e "${TARGET}" ]; then
  if command -v shred >/dev/null 2>&1; then
    find "${TARGET}" -type f -exec shred -u {} + 2>/dev/null || true
  fi
  rm -rf "${TARGET}"
  echo "[teardown] removed ${TARGET}"
fi

# 4. Clear secrets from THIS shell only.
unset WORKER_TOKEN CONTROL_PLANE_URL HF_TOKEN 2>/dev/null || true

echo "[teardown] done — only your footprint removed. Other tenants untouched."
echo "[teardown] NOTE: this is a shared box; there is no instance to terminate."
echo "           Freeing your GPU = your process exiting (done above)."
