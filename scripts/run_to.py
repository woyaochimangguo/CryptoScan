#!/usr/bin/env python3
"""Hard wall-clock timeout wrapper. Usage:
    scripts/run_to.py <seconds> <cmd> [args...]
Exits with the child's exit code, or 124 on timeout (like GNU `timeout`).
Uses os.killpg so children of the child also die."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: run_to.py <seconds> <cmd> [args...]", file=sys.stderr)
        return 2
    secs = float(sys.argv[1])
    cmd = sys.argv[2:]

    proc = subprocess.Popen(cmd, start_new_session=True)
    deadline = time.time() + secs
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            if time.time() >= deadline:
                print(f"[run_to] TIMEOUT after {secs}s — killing pgid {proc.pid}", file=sys.stderr)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return 124
            time.sleep(0.2)
    except KeyboardInterrupt:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        return 130


if __name__ == "__main__":
    sys.exit(main())
