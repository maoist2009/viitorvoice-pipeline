#!/usr/bin/env python3
"""Same pipeline as i16_runner but INT8 activations: the control experiment
that tells us whether the int16 activation path itself is what breaks."""
import json
import os
import sys
import time
import zipfile

import numpy as np
import qai_hub as hub

BB = "/home/runner/bb"
STATE = f"{BB}/q8_state.json"
DONE = ("SUCCESS", "FAILED", "CANCELLED", "ERROR")


def st(**kw):
    d = json.load(open(STATE)) if os.path.exists(STATE) else {}
    d.update(kw)
    json.dump(d, open(STATE, "w"), indent=1)
    return d


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


c = hub.Client()
cmd = sys.argv[1]
stt = json.load(open(STATE)) if os.path.exists(STATE) else {}

if cmd == "quant":
    j = c.submit_quantize_job(c.get_model(stt.get("model", "mnjx44g9y")), c.get_dataset("d7l6e44w7"),
                              weights_dtype=hub.QuantizeDtype.INT8,
                              activations_dtype=hub.QuantizeDtype.INT8,
                              name="fe_bb32_w8a8")
    log(f"QUANTIZE {j.job_id}")
    st(quantize8=j.job_id)
elif cmd == "poll":
    while True:
        s = c.get_job(stt["quantize8"]).get_status()
        log(f"{s.code} {s.message[:160]}")
        if s.code in DONE:
            if s.code == "SUCCESS":
                st(qdq8=c.get_job(stt["quantize8"]).get_target_model().model_id)
            break
        time.sleep(40)
elif cmd == "fetch":
    m = c.get_model(json.load(open(STATE))["qdq8"])
    out = m.download(f"{BB}/fe_q8.zip")
    with zipfile.ZipFile(out) as z:
        z.extractall(BB)
    log("members " + str(z.namelist()))
elif cmd == "check":
    import onnxruntime as ort
    p = sys.argv[2]
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    lm = np.fromfile(f"{BB}/fe_logmel.f32", np.float32).reshape(802, 80)
    vf = np.array([154.0], np.float32)
    got = s.run(None, {"log": lm, "valid_frames": vf})[0].reshape(1024, 200)
    ref = np.load(f"{BB}/feat_ref_24960.npy").reshape(1024, 200)
    a, b = got[:, :39], ref[:, :39]
    log("cos %.7f relL2 %.5f" % (float(np.dot(a.ravel(), b.ravel()) /
        (np.linalg.norm(a) * np.linalg.norm(b))), np.linalg.norm(a - b) / np.linalg.norm(b)))
