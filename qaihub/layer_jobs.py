#!/usr/bin/env python3
"""Compile/link/fetch truncated backbones exported by layer_probe.py.

usage: layer_jobs.py submit <n>... | poll | link <n>... | poll-link | fetch
"""
import json
import os
import shutil
import sys

import qai_hub as hub

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "layer_jobs.json")
OPTS = ("--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type float16 "
        "--qairt_version 2.51")
DEV = "Samsung Galaxy S25 (Family)"


def main():
    cmd = sys.argv[1]
    c = hub.Client()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if cmd == "submit":
        for n in sys.argv[2:]:
            if n in st:
                print(n, "already", st[n]["compile"])
                continue
            src = shutil.copy(f"/data/data/com.termux/files/usr/tmp/llm_{n}l.onnx",
                              os.path.join(HERE, f"llm_{n}l.onnx"))
            j = c.submit_compile_job(model=src, device=hub.Device(DEV),
                                     name=f"llm{n}l_fp16", options=OPTS)
            st[n] = {"compile": j.job_id}
            print(n, "COMPILE", j.job_id, flush=True)
            json.dump(st, open(STATE, "w"), indent=1)
    elif cmd == "poll":
        for n, d in st.items():
            s = c.get_job(d["compile"]).get_status()
            print(n, "compile", s.code)
            if "link" in d:
                print(n, "link", c.get_job(d["link"]).get_status().code)
    elif cmd == "link":
        for n in sys.argv[2:]:
            d = st[n]
            if "link" in d:
                print(n, "already linked", d["link"])
                continue
            j = c.get_job(d["compile"])
            if j.get_status().code != "SUCCESS":
                print(n, "not ready", j.get_status().code)
                continue
            lj = c.submit_link_job(models=j.get_target_model(), device=j.device,
                                   name=f"llm{n}l_link", options="--qairt_version 2.51")
            d["link"] = lj.job_id
            print(n, "LINK", lj.job_id, flush=True)
            json.dump(st, open(STATE, "w"), indent=1)
    elif cmd == "fetch":
        for n, d in st.items():
            if "link" not in d:
                continue
            out = os.path.join(HERE, f"probe_{n}l")
            if os.path.isdir(out) and os.listdir(out):
                print(n, "already fetched")
                continue
            os.makedirs(out, exist_ok=True)
            c.get_job(d["link"]).get_target_model().download(out)
            print(n, "fetched", os.listdir(out), flush=True)


if __name__ == "__main__":
    main()
