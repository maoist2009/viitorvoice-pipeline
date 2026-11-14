#!/usr/bin/env python3
"""Compile + link one graph segment on qai-hub (input specs read from the model).

usage: seg_pipeline.py <onnx> <tag>

State lives in /data/data/com.termux/files/usr/tmp/seg_<tag>_state.json so the
process can be restarted at any point.
"""
import json
import os
import sys
import time

import qai_hub

OUT = "/data/data/com.termux/files/usr/tmp"
DEVICE = "Samsung Galaxy S25 (Family)"
OPTIONS = ("--target_runtime qnn_dlc --truncate_64bit_io "
           "--quantize_full_type float16 --qairt_version 2.51")

ONNX, TAG = sys.argv[1], sys.argv[2]
STATE = f"{OUT}/seg_{TAG}_state.json"
BIN = f"{OUT}/seg_{TAG}.bin"

DT = {1: "float32", 10: "float16", 7: "int64", 6: "int32"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def state():
    try:
        return json.load(open(STATE))
    except Exception:                                   # noqa: BLE001
        return {}


def save(**kw):
    st = state()
    st.update(kw)
    json.dump(st, open(STATE, "w"))
    return st


def devices():
    return qai_hub.get_devices()


def device():
    devs = devices()
    return next(d for d in devs if d.name == DEVICE)


def specs(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    out = {}
    for t in m.graph.input:
        tt = t.type.tensor_type
        dims = tuple(d.dim_value for d in tt.shape.dim)
        out[t.name] = (dims, DT.get(tt.elem_type, "float32"))
    return out


def main():
    st = state()
    if "compile_job" not in st:
        sp = specs(ONNX)
        log(f"uploading {ONNX} ({os.path.getsize(ONNX) / 1e6:.1f} MB) specs {sp}")
        job = qai_hub.submit_compile_job(model=ONNX, device=device(),
                                         input_specs=sp, options=OPTIONS)
        save(compile_job=job.job_id)
        log(f"compile job {job.job_id}")
        st = state()

    cid = st["compile_job"]
    while True:
        s = qai_hub.get_job(cid).get_status()
        if str(s.state) in ("State.SUCCESS", "State.FAILED", "State.CANCELLED"):
            break
        log(f"compile {s.state}")
        time.sleep(60)
    log(f"compile -> {s.state}")
    if "SUCCESS" not in str(s.state):
        log(f"FAILED: {getattr(s, 'message', '')[:200]}")
        return 1

    if "link_job" not in st:
        model = qai_hub.get_job(cid).get_target_model()
        lj = qai_hub.submit_link_job(models=[model], device=device(),
                                     options="--qairt_version 2.51")
        save(link_job=lj.job_id)
        log(f"link job {lj.job_id}")
        st = state()

    lid = st["link_job"]
    while True:
        s = qai_hub.get_job(lid).get_status()
        if str(s.state) in ("State.SUCCESS", "State.FAILED", "State.CANCELLED"):
            break
        log(f"link {s.state}")
        time.sleep(60)
    log(f"link -> {s.state}")
    if "SUCCESS" not in str(s.state):
        st = state()
        st.pop("link_job", None)          # a failed link must be re-submitted
        json.dump(st, open(STATE, "w"))
        try:
            qai_hub.get_job(lid).download_artifacts_for_type(
                OUT, qai_hub.JobArtifactType.HUB_LOG)
        except Exception:                               # noqa: BLE001
            pass
        log(f"FAILED: {getattr(s, 'message', '')[:200]}")
        return 2

    got = qai_hub.get_job(lid).download_target_model(BIN)
    if got and os.path.abspath(got) != os.path.abspath(BIN):
        import shutil
        shutil.move(got, BIN)
    log(f"bin {BIN} {os.path.getsize(BIN) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
