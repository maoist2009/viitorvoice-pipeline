#!/usr/bin/env python3
"""w8a16 rebuild driven entirely from the phone.

The cloud runner is gone, but everything needed is already on qai-hub: each
calibrated w8a8 compile job carries its source model (`job.model`) and its
uploaded calibration set (`job.calibration_dataset`). Submitting a new compile
with those two objects uploads nothing, so the whole a16 pass costs the phone
only the resulting bins.

usage: a16_from_phone.py submit | poll | link | fetch
"""
import json
import os
import sys

import qai_hub as hub

OPTS = ("--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type w8a16 "
        "--qairt_version 2.51")
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a16_jobs.json")
SRC = {
    "p2": "jgjyvky7g",
    "p3": "jpexd4x7g",
    "p4": "jgzy3vyzp",
    "p5chunk48": "jg92l82qp",
}
DONE = ("SUCCESS", "FAILED", "CANCELLED", "ERROR")


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(d):
    json.dump(d, open(STATE, "w"), indent=1)


def main():
    cmd = sys.argv[1]
    c = hub.Client()
    st = load()

    if cmd == "submit":
        for name, jid in SRC.items():
            src = c.get_job(jid)
            j = c.submit_compile_job(model=src.model, device=src.device,
                                     name=f"{name}_a16", options=OPTS,
                                     calibration_data=src.calibration_dataset)
            st.setdefault("compile", {})[name] = j.job_id
            print(name, "COMPILE", j.job_id, flush=True)
        save(st)

    elif cmd == "poll":
        for name, jid in st.get("compile", {}).items():
            s = c.get_job(jid).get_status()
            print(f"{name:<11} {s.code:<18} {s.message[:120]}", flush=True)

    elif cmd == "link":
        st.setdefault("link", {})
        for name, jid in st.get("compile", {}).items():
            j = c.get_job(jid)
            if j.get_status().code != "SUCCESS" or name in st["link"]:
                continue
            lj = c.submit_link_job(models=j.get_target_model(), device=j.device,
                                   name=f"{name}_a16_link", options="--qairt_version 2.51")
            st["link"][name] = lj.job_id
            print(name, "LINK", lj.job_id, flush=True)
        save(st)

    elif cmd == "poll-link":
        for name, jid in st.get("link", {}).items():
            s = c.get_job(jid).get_status()
            print(f"{name:<11} {s.code:<18} {s.message[:120]}", flush=True)

    elif cmd == "fetch":
        out = os.path.join(os.path.dirname(STATE), "dec_split", "a16")
        os.makedirs(out, exist_ok=True)
        for name, jid in st.get("link", {}).items():
            j = c.get_job(jid)
            if j.get_status().code != "SUCCESS":
                print(name, "skip (not success)")
                continue
            path = os.path.join(out, f"{name}.bin")
            j.get_target_model().download(path)
            print(name, os.path.getsize(path) // 1048576, "MB", flush=True)

    else:
        raise SystemExit(__doc__)


main()
