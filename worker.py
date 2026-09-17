"""
STUDIO GPU worker — runs on the L40S box.

Loop: claim a queued job from the control plane -> run it on the GPU ->
report status back. Outputs land in ./outputs/ for MANUAL download.
The worker never contacts your platform. When the queue drains, it exits
(so the caller can tear the whole box down and stop paying).

Secrets: CONTROL_PLANE_URL + WORKER_TOKEN come from the environment only.
Nothing is written to disk except the generated media in ./outputs/.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = os.environ["CONTROL_PLANE_URL"].rstrip("/")
TOKEN = os.environ["WORKER_TOKEN"]
IDLE_EXITS_AFTER = int(os.environ.get("IDLE_EXITS_AFTER", "3"))  # empty polls before exit
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "5"))
OUT = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
OUT.mkdir(parents=True, exist_ok=True)

H = {"Authorization": f"Bearer {TOKEN}"}

from run_job import run_job  # noqa: E402


def claim():
    r = requests.post(f"{BASE}/jobs/claim", headers=H, timeout=30)
    r.raise_for_status()
    return r.json()["job"]


def report(job_id, status, artifact_ref=None, error=None):
    requests.post(f"{BASE}/jobs/{job_id}/status", headers=H, timeout=30,
                  json={"status": status, "artifact_ref": artifact_ref, "error": error})


def main():
    empty = 0
    print(f"[worker] polling {BASE}")
    while True:
        try:
            job = claim()
        except Exception as e:
            print(f"[worker] claim error: {e}", file=sys.stderr)
            time.sleep(POLL_SECONDS)
            continue

        if job is None:
            empty += 1
            print(f"[worker] queue empty ({empty}/{IDLE_EXITS_AFTER})")
            if empty >= IDLE_EXITS_AFTER:
                print("[worker] idle — exiting so the box can be torn down.")
                return
            time.sleep(POLL_SECONDS)
            continue

        empty = 0
        jid = job["id"]
        print(f"[worker] job {jid}: {job['channel']}/{job['kind']}")
        report(jid, "running")
        try:
            payload = json.loads(job["payload"])
            out_path = run_job(job["channel"], job["kind"], payload, OUT, jid)
            # artifact_ref is just a NOTE of the local filename — the file itself
            # stays on the worker for you to download manually (air-gap).
            report(jid, "done", artifact_ref=out_path.name)
            print(f"[worker] job {jid} done -> {out_path}")
        except Exception as e:
            report(jid, "failed", error=str(e)[:500])
            print(f"[worker] job {jid} FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
