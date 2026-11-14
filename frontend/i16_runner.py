#!/usr/bin/env python3
"""fp32-source INT16-activation quantisation, driven from the runner.

The fp16 source cannot be quantised (the analytic bias scale falls under the
fp16 floor), which is why every earlier w8a16/int16 attempt produced garbage.
This uses the freshly exported fp32 graph and the hub's real quantize job, then
checks the QDQ output in ONNX Runtime before paying for a compile.

usage: i16.py upload | quant | poll | qc-fetch | ortcheck | compile | link | fetch
"""
import json
import os
import sys
import time

import numpy as np

import qai_hub as hub

BB = "/home/runner/bb"
STATE = f"{BB}/i16_state.json"
QDQ = os.environ.get("QDQ", f"{BB}/fe_qdq.onnx")
BIN = f"{BB}/fe_i16.bin"
DATASET = os.environ.get("DATASET", "d7l6e44w7")
DONE = ("SUCCESS", "FAILED", "CANCELLED", "ERROR")


def st_read():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def st_write(**kw):
    st = st_read()
    st.update(kw)
    json.dump(st, open(STATE, "w"), indent=1)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wait(job_id, label):
    while True:
        s = hub.get_job(job_id).get_status()
        log(f"{label} {s.code} {s.message[:200]}")
        if s.code in DONE:
            return s.code
        time.sleep(40)


def main():
    cmd = sys.argv[1]
    c = hub.Client()
    st = st_read()

    if cmd == "upload":
        m = c._upload_model_if_needed(f"{BB}/pub", model_type=hub.SourceModelType.ONNX,
                                     name="fe_bb32_clean")
        st_write(model=m.model_id)
        log(f"MODEL {m.model_id}")

    elif cmd == "quant":
        j = c.submit_quantize_job(c.get_model(st["model"]),
                                  c.get_dataset(DATASET),
                                  weights_dtype=hub.QuantizeDtype.INT8,
                                  activations_dtype=hub.QuantizeDtype.INT16,
                                  name="fe_bb32_w8a16")
        st_write(quantize=j.job_id)
        log(f"QUANTIZE {j.job_id}")

    elif cmd == "poll":
        code = wait(st["quantize"], "quantize")
        if code == "SUCCESS":
            j = c.get_job(st["quantize"])
            st_write(qdq=j.get_target_model().model_id)
            log(f"QDQ {st_read()['qdq']}")

    elif cmd == "qc-fetch":
        m = c.get_model(st["qdq"])
        log(f"downloading {m.model_id} {m.name}")
        out = m.download(f"{BB}/fe_qdq.zip")
        log(f"saved {out}")
        import zipfile
        with zipfile.ZipFile(out) as z:
            log("members " + str(z.namelist())[:200])
            z.extractall(BB)
        log("files " + str([f for f in os.listdir(BB) if f.endswith(".onnx")]))


    elif cmd == "ortcheck":
        import onnxruntime as ort
        s = ort.InferenceSession(QDQ, providers=["CPUExecutionProvider"])
        log(f"inputs {[i.name for i in s.get_inputs()]}")
        logmel = np.fromfile(f"{BB}/fe_logmel.f32", np.float32).reshape(802, 80)
        vf = np.array([154.0], np.float32)
        got = s.run(None, {"log": logmel, "valid_frames": vf})[0].reshape(1024, 200)
        ref = np.load(f"{BB}/feat_ref_24960.npy").reshape(1024, 200)
        a, b = got[:, :39], ref[:, :39]
        log("QDQ vs fp32 ref: cos %.8f relL2 %.6f maxabs %.4f" %
            (float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b))),
             np.linalg.norm(a - b) / np.linalg.norm(b), np.abs(a - b).max()))

    elif cmd == "compile":
        opts = ("--target_runtime qnn_dlc --truncate_64bit_io "
                "--qairt_version 2.51")
        j = c.submit_compile_job(model=c.get_model(st["qdq"]),
                                 device=hub.Device("Samsung Galaxy S25 (Family)"),
                                 name="fe_bb32_i16", options=opts)
        st_write(compile=j.job_id)
        log(f"COMPILE {j.job_id}")

    elif cmd == "link":
        j = c.get_job(st["compile"])
        lj = c.submit_link_job(models=[j.get_target_model()], device=j.device,
                               name="fe_bb32_i16_link",
                               options="--qairt_version 2.51")
        st_write(link=lj.job_id)
        log(f"LINK {lj.job_id}")

    elif cmd == "fetch":
        m = c.get_job(st["link"]).get_target_model()
        out = m.download()
        log(f"BIN {out} {os.path.getsize(out)}")


if __name__ == "__main__":
    main()
