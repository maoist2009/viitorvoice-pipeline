#!/usr/bin/env python3
"""Upload the fp32 w2v backbone as an external-data ONNX directory.

The fp32 graph is 2.25 GB, well past protobuf's 2 GiB ceiling, so it travels as
`fe_backbone.onnx` + `fe_backbone.onnx.data`; qai_hub zips the directory itself
and unzips it server side.  The two members are hard links, so this costs no
extra disk.
"""
import itertools
import os
import sys
import time

sys.path.insert(0, "/data/data/com.termux/files/home/projects/rvc/rvc-android/salvage/dl_vv")
import pin_vv  # noqa: E402
import qai_hub as hub  # noqa: E402

import json
PINS = json.load(open("/data/data/com.termux/files/usr/tmp/pins.json"))
for h, ips in PINS.items():
    pin_vv.HOSTS[h] = tuple(ips)
    pin_vv._cycle[h] = itertools.cycle(ips)
    print(h, ips, flush=True)

SRC = "/data/data/com.termux/files/home/tmp/fe_npu"
PUB = "/data/data/com.termux/files/usr/tmp/bb32pub"

os.makedirs(PUB, exist_ok=True)
for f in ("fe_backbone.onnx", "fe_backbone.onnx.data"):
    dst = os.path.join(PUB, f)
    if not os.path.exists(dst):
        import shutil
        shutil.copyfile(os.path.join(SRC, f), dst)
print("publish dir", os.listdir(PUB), flush=True)

t0 = time.time()
c = hub.Client()
m = c._upload_model_if_needed(PUB, model_type=hub.SourceModelType.ONNX,
                            name="fe_backbone_fp32")
print("model", m.model_id, "%.0f s" % (time.time() - t0), flush=True)
print("MODEL_ID=" + m.model_id, flush=True)
