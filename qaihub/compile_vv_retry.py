#!/usr/bin/env python3
"""compile_vv.py with whole-submission retry for flaky S3 accelerate uploads."""
import subprocess, sys, time, os
CMD = [sys.executable, "-u", "/data/data/com.termux/files/home/tmp/dl_vv/compile_vv.py"]
LOG = "/data/data/com.termux/files/home/tmp/dl_vv/retry.log"
BIN = "/data/data/com.termux/files/home/tmp/dl_vv/dualcodec_decoder.bin"

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "decoder"
    for attempt in range(1, 9):
        with open(LOG, "a") as f:
            f.write(f"\n===== attempt {attempt} {time.strftime('%H:%M:%S')} =====\n")
        r = subprocess.run(CMD + [which], capture_output=True, text=True, timeout=3600)
        with open(LOG, "a") as f:
            f.write(r.stdout[-6000:] + r.stderr[-6000:] + f"\n[rc={r.returncode}]\n")
        if os.path.isfile(BIN) and os.path.getsize(BIN) > 1_000_000:
            print(f"attempt {attempt}: BIN OK", flush=True); return 0
        print(f"attempt {attempt} rc={r.returncode}; retry in 30s", flush=True)
        time.sleep(30)
    return 1

if __name__ == "__main__":
    sys.exit(main())
