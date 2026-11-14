#!/usr/bin/env python3
"""w8a16 compile of the w2v backbone, driven from the phone with no big upload.

The fp16 source model is already on qai-hub (compile job SRC); only the ~1 MB
calibration set has to go up.

usage: fe_a16.py submit | poll | link | fetch
"""
import json
import os
import sys

import numpy as np

import pin_vv  # noqa: F401  (real IPs for workbench + S3)
import qai_hub as hub

OPTS = ("--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type w8a16 "
        "--qairt_version 2.51")
SRC = "jgd4l8zzg"                       # fp16 compile of fe_bb16.onnx
STATE = os.environ.get("STATE","/data/data/com.termux/files/home/tmp/fe_npu/fe_a16_state.json")
CAL = os.environ.get("CAL","/data/data/com.termux/files/home/tmp/fe_npu/cal_data.npz")
OUT = "/data/data/com.termux/files/usr/tmp"
BIN = os.environ.get("BIN", f"{OUT}/seg_fea16.bin")
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
        src = c.get_job(SRC)
        z = np.load(CAL)
        dt = np.float32 if os.environ.get("CAL_DT") == "f32" else np.float16
        data = {"log": [z["log"][i].astype(dt) for i in range(z["log"].shape[0])],
                "valid_frames": [z["vf"][i].astype(dt) for i in range(z["vf"].shape[0])]}
        ds = c.upload_dataset(data, name="fe_cal_5x")
        print("dataset", ds.dataset_id, flush=True)
        j = c.submit_compile_job(model=src.model, device=src.device,
                                 name="fe_bb_w8a16", options=OPTS,
                                 calibration_data=ds)
        st["dataset"] = ds.dataset_id
        st["compile"] = j.job_id
        save(st)
        print("COMPILE", j.job_id, flush=True)

    elif cmd == "poll":
        s = c.get_job(st["compile"]).get_status()
        print(st["compile"], s.code, s.message[:200], flush=True)

    elif cmd == "link":
        j = c.get_job(st["compile"])
        m = j.get_target_model()
        lj = c.submit_link_job(models=[m], device=j.device,
                              name="fe_bb_w8a16_link",
                              options="--qairt_version 2.51")
        st["link"] = lj.job_id
        save(st)
        print("LINK", lj.job_id, flush=True)

    elif cmd == "fetch":
        j = c.get_job(st["link"])
        got = j.download_target_model(BIN)
        if got and os.path.abspath(got) != os.path.abspath(BIN):
            import shutil
            shutil.move(got, BIN)
        print("bin", BIN, os.path.getsize(BIN) / 1e6, "MB", flush=True)


if __name__ == "__main__":
    main()
