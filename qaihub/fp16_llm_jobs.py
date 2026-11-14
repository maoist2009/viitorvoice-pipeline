#!/usr/bin/env python3
"""Compile the llm backbone in fp16 (no int8) from the already-uploaded pt2.

The w8a16 graph disagrees with the torch reference (per-token cos 0.14 for the
same inputs_embeds), so this isolates quantization from the export itself.
Nothing is uploaded: the source CompileJob j57o7d8vg still carries the model.

usage: fp16_llm_jobs.py submit|poll|link|poll-link|fetch
"""
import json
import os
import sys

import qai_hub as hub

OPTS = ("--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type float16 "
        "--qairt_version 2.51")
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp16_llm_jobs.json")
SRC = "j57o7d8vg"


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(d):
    json.dump(d, open(STATE, "w"), indent=1)


def main():
    cmd = sys.argv[1]
    c = hub.Client()
    st = load()
    if cmd == "submit":
        if "compile" in st:
            print("already", st["compile"])
        else:
            src = c.get_job(SRC)
            j = c.submit_compile_job(model=src.model, device=src.device,
                                     name="llm_fp16", options=OPTS)
            st["compile"] = j.job_id
            save(st)
            print("COMPILE", j.job_id, flush=True)
    elif cmd == "poll":
        s = c.get_job(st["compile"]).get_status()
        print(f"{s.code:<18} {s.message[:160]}", flush=True)
    elif cmd == "link":
        if "link" in st:
            print("already", st["link"])
            return
        j = c.get_job(st["compile"])
        if j.get_status().code != "SUCCESS":
            print("not ready:", j.get_status().code)
            return
        lj = c.submit_link_job(models=j.get_target_model(), device=j.device,
                               name="llm_fp16_link", options="--qairt_version 2.51")
        st["link"] = lj.job_id
        save(st)
        print("LINK", lj.job_id, flush=True)
    elif cmd == "poll-link":
        s = c.get_job(st["link"]).get_status()
        print(f"{s.code:<18} {s.message[:160]}", flush=True)
    elif cmd == "fetch":
        j = c.get_job(st["link"])
        m = j.get_target_model()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp16")
        os.makedirs(out, exist_ok=True)
        m.download(out)
        print("downloaded to", out, os.listdir(out))
    save(st)


if __name__ == "__main__":
    main()
