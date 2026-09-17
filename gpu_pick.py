"""
Pick a free GPU on the shared box and return its UUID.

Pinning by UUID (not index) is mandatory here: GPU 5 is broken and not enumerated
by CUDA, so CUDA ordinals are shifted vs nvidia-smi indices. UUID sidesteps that.

Usage:
    export CUDA_VISIBLE_DEVICES="$(python3 gpu_pick.py --need-gb 40)"
Prints the UUID of a GPU with enough free memory, or exits non-zero.
Preference order matches the operator's free-list (idle first, avoid MPS).
"""
import argparse
import subprocess
import sys

# Operator-supplied free-list, best first. Avoids GPU 5 (broken) and busy cards.
PREFERRED = [
    "GPU-8b870518-304a-92a4-b17e-c2633fca54e1",  # smi 1: idle, ~44GB
    "GPU-c66ad1df-9b90-95d8-c204-e8e55371427f",  # smi 0: ~42GB, only VST
    "GPU-0a007bf8-73ae-c63e-8a73-e646d051e4e2",  # smi 7: ~36GB but MPS active
    "GPU-857126e7-20d3-1d69-c1c7-5e1960f20309",  # smi 2: ~32GB
]


def free_by_uuid():
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=uuid,memory.free",
         "--format=csv,noheader,nounits"], text=True)
    d = {}
    for line in out.strip().splitlines():
        uuid, free = [x.strip() for x in line.split(",")]
        d[uuid] = int(free)  # MiB
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-gb", type=float, default=40.0)
    a = ap.parse_args()
    need_mib = a.need_gb * 1024
    live = free_by_uuid()

    # Try preferred order first, then any GPU with enough free memory.
    for uuid in PREFERRED + list(live.keys()):
        if live.get(uuid, 0) >= need_mib:
            print(uuid)
            return
    sys.stderr.write(
        f"[gpu_pick] no GPU with >= {a.need_gb}GB free. Current free (MiB): {live}\n")
    sys.exit(1)


if __name__ == "__main__":
    main()
