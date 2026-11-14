#!/usr/bin/env python3
"""Quantise the fp32 w2v backbone with INT16 activations, hub side.

The compile-time `--quantize_full_type w8a16` shortcut gave cos 0.40, so this
uses the real quantize job: weights INT8 / activations INT16 on the fp32 source
(the fp16 source cannot give the quantiser meaningful ranges).  Its QDQ output
is downloaded and checked in ONNX Runtime before anything is compiled.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/data/data/com.termux/files/home/projects/rvc/rvc-android/salvage/dl_vv")
import pin_vv  # noqa: E402
import qai_hub as hub  # noqa: E402

EDGES = ["13.35.221.233", "13.35.213.211", "13.226.64.144",
         "13.249.36.38", "13.226.71.165"]
API = ["13.35.190.35", "13.35.190.127", "13.35.190.115"]
for h in ("workbench.aihub.qualcomm.com",):
    pin_vv.HOSTS[h] = tuple(API)
    pin_vv._cycle[h] = itertools.cycle(API)
for h in ("tetrahub-qprod-userdata.s3-accelerate.amazonaws.com",
          "tetrahub-qprod-userdata.s3.amazonaws.com"):
    pin_vv.HOSTS[h] = tuple(EDGES)
    pin_vv._cycle[h] = itertools.cycle(EDGES)

STATE = "/data/data/com.termux/files/home/tmp/fe_npu/fe_q16_state.json"
CAL = "/data/data/com.termux/files/home/tmp/fe_npu/cal_data15.npz"
MODEL = os.environ.get("MODEL_ID", "mnwee4673")

st = json.load(open(STATE)) if os.path.exists(STATE) else {}
z = np.load(CAL)
data = {"log": [z["log"][i].astype(np.float32) for i in range(z["log"].shape[0])],
        "valid_frames": [z["vf"][i].astype(np.float32) for i in range(z["vf"].shape[0])]}
print("calib", len(data["log"]), "vf", [int(v[0]) for v in data["valid_frames"]], flush=True)

c = hub.Client()
ds = c.upload_dataset(data, name="fe_cal15_f32")
st["dataset"] = ds.dataset_id
print("dataset", ds.dataset_id, flush=True)

job = c.submit_quantize_job(
    c.get_model(MODEL),
    ds,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT16,
    name="fe_bb_w8a16_fp32src",
)
st["quantize"] = job.job_id
json.dump(st, open(STATE, "w"))
print("QUANTIZE", job.job_id, flush=True)
