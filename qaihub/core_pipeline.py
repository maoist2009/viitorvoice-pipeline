#!/usr/bin/env python3
"""Compile + link the dualcodec encode core on qai-hub, network-flake tolerant.

usage: core_pipeline.py <onnx> <samples> <frames> <tag>

State is persisted in /data/data/com.termux/files/usr/tmp/core_<tag>_state.json so
the pipeline can be restarted at any point without repeating finished steps.
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

ONNX, N, F, TAG = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
STATE = f"{OUT}/core_{TAG}_state.json"
DLC = f"{OUT}/encode_core_{TAG}.dlc"
BIN = f"{OUT}/encode_core_{TAG}.bin"


def io_dtype(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    t = {v.name: v.type.tensor_type.elem_type for v in m.graph.input}
    names = {10: "float16", 1: "float32"}
    d = names.get(t.get("audio", 1), "float32")
    return d, names.get(t.get("semantic_repr", 1), "float32")


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


def retry(fn, what, n=1, delay=5):
    for i in range(n):
        try:
            return fn()
        except Exception as e:                          # noqa: BLE001
            log(f"{what} failed ({i + 1}/{n}): {str(e)[:140]}")
            time.sleep(delay)
    raise SystemExit(f"{what} giving up")


def device():
    devs = retry(lambda: qai_hub.get_devices(), "get_devices")
    return next(d for d in devs if d.name == DEVICE)


def do_compile():
    dev = device()
    da, ds = io_dtype(ONNX)
    log(f"uploading {ONNX} ({os.path.getsize(ONNX) / 1e6:.1f} MB) "
        f"specs audio(1,1,{N}) {da} semantic_repr(1,1024,{F}) {ds}")
    job = retry(lambda: qai_hub.submit_compile_job(
        model=ONNX, device=dev,
        input_specs={"audio": ((1, 1, N), da),
                     "semantic_repr": ((1, 1024, F), ds)},
        options=OPTIONS), "submit_compile")
    save(compile_job=job.job_id)
    log(f"compile job {job.job_id}")
    return job


def wait(job_id, what):
    while True:
        st = retry(lambda: qai_hub.get_job(job_id).get_status(), f"{what} status")
        if str(st.state) in ("State.SUCCESS", "State.FAILED", "State.CANCELLED"):
            return st
        log(f"{what} {st.state}")
        time.sleep(60)


def main():
    st = state()
    if "compile_job" not in st:
        do_compile()
        st = state()
    cid = st["compile_job"]
    log(f"waiting compile {cid}")
    cst = wait(cid, "compile")
    log(f"compile -> {cst.state}")
    if "SUCCESS" not in str(cst.state):
        log(f"compile failed: {getattr(cst, 'message', '')[:200]}")
        return 1

    if "link_job" not in st:
        job = qai_hub.get_job(cid)
        model = retry(job.get_target_model, "get_target_model")
        dev = device()
        lj = retry(lambda: qai_hub.submit_link_job(
            models=[model], device=dev, options="--qairt_version 2.51"),
            "submit_link")
        save(link_job=lj.job_id)
        log(f"link job {lj.job_id}")
        st = state()

    lid = st["link_job"]
    log(f"waiting link {lid}")
    lst = wait(lid, "link")
    log(f"link -> {lst.state}")
    if "SUCCESS" not in str(lst.state):
        try:
            retry(lambda: qai_hub.get_job(lid).download_artifacts_for_type(
                OUT, qai_hub.JobArtifactType.HUB_LOG), "hub log")
        except SystemExit:
            pass
        log(f"link failed: {getattr(lst, 'message', '')[:200]}")
        return 2

    got = retry(lambda: qai_hub.get_job(lid).download_target_model(BIN),
                "download bin")
    if got and os.path.abspath(got) != os.path.abspath(BIN):
        import shutil
        shutil.move(got, BIN)
    log(f"bin {BIN} {os.path.getsize(BIN) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
