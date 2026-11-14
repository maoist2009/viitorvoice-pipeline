#!/usr/bin/env python3
"""Download the linked llm_theta1m context binary straight from qai-hub (S3)."""
import os
import sys

import qai_hub as hub

JOB = sys.argv[1] if len(sys.argv) > 1 else "jpvmmewmp"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/data/data/com.termux/files/usr/tmp/vv_nar/bin1m"


def main():
    os.makedirs(OUT, exist_ok=True)
    j = hub.get_job(JOB)
    print("status", j.get_status().code, flush=True)
    m = j.get_target_model()
    print("model", m.model_id, getattr(m, "name", ""), flush=True)
    m.download(OUT)
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f"{os.path.getsize(p)/1e6:10.1f} MB  {f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
