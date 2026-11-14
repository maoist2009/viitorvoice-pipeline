#!/usr/bin/env python3
"""Compile the tiny mask-probe ONNX and check on device whether the compiled
graph still honours attention_mask.

usage: mask_probe_job.py submit|poll|link|poll-link|fetch
"""
import json
import os
import shutil
import sys

import qai_hub as hub

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = "/data/data/com.termux/files/usr/tmp/mask_probe.onnx"
STATE = os.path.join(HERE, "mask_probe_job.json")
OPTS = ("--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type w8a16 "
        "--qairt_version 2.51")


def main():
    cmd = sys.argv[1]
    c = hub.Client()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if cmd == "submit":
        src = shutil.copy(ONNX, os.path.join(HERE, "mask_probe.onnx"))
        j = c.submit_compile_job(model=src, device=hub.Device(HUB_DEVICE),
                                 name="mask_probe", options=OPTS)
        st["compile"] = j.job_id
        print("COMPILE", j.job_id)
    elif cmd == "poll":
        s = c.get_job(st["compile"]).get_status()
        print(s.code, s.message[:200])
    elif cmd == "link":
        j = c.get_job(st["compile"])
        lj = c.submit_link_job(models=j.get_target_model(), device=j.device,
                               name="mask_probe_link", options="--qairt_version 2.51")
        st["link"] = lj.job_id
        print("LINK", lj.job_id)
    elif cmd == "poll-link":
        s = c.get_job(st["link"]).get_status()
        print(s.code, s.message[:200])
    elif cmd == "fetch":
        out = os.path.join(HERE, "mask_probe_bin")
        os.makedirs(out, exist_ok=True)
        c.get_job(st["link"]).get_target_model().download(out)
        print(os.listdir(out))
    json.dump(st, open(STATE, "w"), indent=1)


HUB_DEVICE = "Samsung Galaxy S25 (Family)"

if __name__ == "__main__":
    main()
