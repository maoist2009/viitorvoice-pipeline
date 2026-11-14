#!/usr/bin/env python3
"""Compile the fixed (rope_theta=1e6) LLM backbone ONNX to a QNN DLC, fp16.

The previously compiled llm.bin baked rope_theta=10000 because the exporter read
config.json's `rope_parameters` dict while transformers 4.57 uses the legacy
`rope_theta` attribute (default 10000).  See export_theta_check.py: with the
base pinned to 1e6 the ORT graph matches torch bit-for-bit (cos 1.000000).

usage: llm1m_jobs.py submit|poll|link|poll-link|fetch
"""
import json
import os
import sys

import qai_hub as hub

OPTS = ("--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type float16 "
        "--qairt_version 2.51")
DEVICE = "Samsung Galaxy S25 (Family)"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "llm1m_jobs.json")
ONNX = os.path.join(HERE, "llm_theta1m.onnx")


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(d):
    json.dump(d, open(STATE, "w"), indent=1)


def main():
    cmd = sys.argv[1]
    c = hub.Client()
    st = load()
    if cmd == "submit":
        if "upload" in st:
            print("already uploaded", st["upload"])
        else:
            m = c.upload_model(ONNX)
            st["upload"] = m.model_id
            save(st)
            print("UPLOAD", m.model_id, flush=True)
        if "compile" not in st:
            j = c.submit_compile_job(model=hub.get_model(st["upload"]),
                                     device=hub.Device(DEVICE),
                                     name="llm_fp16_theta1m", options=OPTS)
            st["compile"] = j.job_id
            save(st)
            print("COMPILE", j.job_id, flush=True)
    elif cmd == "poll":
        s = c.get_job(st["compile"]).get_status()
        print(f"{s.code:<18} {s.message[:200]}", flush=True)
    elif cmd == "link":
        if "link" in st:
            print("already", st["link"])
            return
        j = c.get_job(st["compile"])
        if j.get_status().code != "SUCCESS":
            print("not ready:", j.get_status().code)
            return
        lj = c.submit_link_job(models=j.get_target_model(), device=j.device,
                               name="llm_fp16_theta1m_link",
                               options="--qairt_version 2.51")
        st["link"] = lj.job_id
        save(st)
        print("LINK", lj.job_id, flush=True)
    elif cmd == "poll-link":
        s = c.get_job(st["link"]).get_status()
        print(f"{s.code:<18} {s.message[:200]}", flush=True)
    elif cmd == "fetch":
        j = c.get_job(st["link"])
        m = j.get_target_model()
        out = os.path.join(HERE, "bin1m")
        os.makedirs(out, exist_ok=True)
        m.download(out)
        print("downloaded to", out, os.listdir(out))
    save(st)


if __name__ == "__main__":
    main()
