#!/usr/bin/env bash
# STUDIO teardown — cost + secret + privacy hygiene.
# Removes the scratch workdir (incl. outputs you should have already downloaded),
# clears the cloned repo if asked, and prunes docker/k8s so nothing keeps billing.
#
# This is resource cleanup, NOT log-tampering: it does not touch the cloud
# provider's own audit trail, and it is not designed to defeat forensics.
#
# Usage:
#   ./teardown.sh /tmp/studio.XXXXXX [--repo /path/to/clone] [--pod <name>] [--k8s-job <name>]
set -euo pipefail

WORKDIR="${1:?pass the workdir from bootstrap}"; shift || true
REPO=""; POD=""; K8S_JOB=""
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2;;
    --pod) POD="$2"; shift 2;;
    --k8s-job) K8S_JOB="$2"; shift 2;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done

secure_rm() {
  local target="$1"
  [ -e "$target" ] || return 0
  if command -v shred >/dev/null 2>&1; then
    find "$target" -type f -exec shred -u {} + 2>/dev/null || true
  fi
  rm -rf "$target"
  echo "[teardown] removed ${target}"
}

# 1. Scratch workdir (venv, worker scripts, outputs, any temp media)
secure_rm "${WORKDIR}"

# 2. The cloned repo, if you want it gone from this box
[ -n "${REPO}" ] && secure_rm "${REPO}"

# 3. Docker: stop worker container + reclaim all unused resources (stops billing)
if command -v docker >/dev/null 2>&1; then
  [ -n "${POD}" ] && docker rm -f "${POD}" 2>/dev/null || true
  docker system prune -af --volumes >/dev/null 2>&1 || true
  echo "[teardown] docker pruned"
fi

# 4. Kubernetes: delete the ephemeral GPU Job (releases the node)
if command -v kubectl >/dev/null 2>&1 && [ -n "${K8S_JOB}" ]; then
  kubectl delete job "${K8S_JOB}" --ignore-not-found >/dev/null 2>&1 || true
  echo "[teardown] k8s job ${K8S_JOB} deleted"
fi

# 5. Clear this shell's secrets from the current session env
unset WORKER_TOKEN CONTROL_PLANE_URL 2>/dev/null || true

echo "[teardown] done. Nothing left running; secrets cleared from env."
echo "[teardown] Reminder: also STOP/TERMINATE the GPU instance in the provider console"
echo "           so you stop paying — that is the step that actually ends billing."
