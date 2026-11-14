#!/usr/bin/env python3
"""fp32-source INT16-activation quantisation pipeline for the w2v backbone.

Why this exists: `--quantize_full_type w8a16` applied to the *fp16* model gave
cos 0.40 against fp32, because the quantiser derives its ranges from weights
that are already rounded.  Here the source is the fp32 graph (uploaded as
`fe_backbone.onnx` + `.data`, model mnwee4673) and the quantiser runs as its own
hub job, so the QDQ result can be downloaded and checked in ONNX Runtime before
a single compile is paid for.

steps
  quant     upload fp32 calibration, submit the INT8-w / INT16-a quantize job
  qc-poll   wait for the quantize job
  qc-fetch  download the QDQ onnx
  ortcheck  run the QDQ model next to the fp32 reference (cos / relL2 / codes)
  compile   compile the QDQ model for the HTP
  link      link the compiled graph into one context binary
  fetch     download the .bin
"""
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/data/data/com.termux/files/home/projects/rvc/rvc-android/salvage/dl_vv")
import pin_vv  # noqa: E402
import qai_hub as hub  # noqa: E402

HOME = "/data/data/com.termux/files/home/tmp/fe_npu"
TMP = "/data/data/com.termux/files/usr/tmp"
STATE = f"{HOME}/fe_i16_state.json"
PINS = "/data/data/com.termux/files/usr/tmp/pins.json"
SRC_MODEL = os.environ.get("SRC_MODEL", "mnwee4673")   # fp32 backbone
CAL = f"{HOME}/cal_data15_f32.npz"
QDQ = f"{HOME}/fe_qdq.onnx"
BIN = f"{TMP}/seg_fe_i16.bin"
HOST_S3 = "tetrahub-qprod-userdata.s3-accelerate.amazonaws.com"
DONE = ("SUCCESS", "FAILED", "CANCELLED", "ERROR")


def pins():
    if os.path.exists(PINS):
        p = json.load(open(PINS))
    else:
        p = {}
    if "workbench.aihub.qualcomm.com" in p:
        api = tuple(p["workbench.aihub.qualcomm.com"])
        pin_vv.HOSTS["workbench.aihub.qualcomm.com"] = api
        pin_vv._cycle["workbench.aihub.qualcomm.com"] = itertools.cycle(api)
    if HOST_S3 in p:
        s3 = tuple(p[HOST_S3])
        pin_vv.HOSTS[HOST_S3] = s3
        pin_vv._cycle[HOST_S3] = itertools.cycle(s3)
    return p


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
        try:
            s = hub.get_job(job_id).get_status()
            if s.code in DONE:
                log(f"{label} {s.code}: {s.message[:200]}")
                return s.code
        except Exception as e:                       # noqa: BLE001
            log(f"{label} poll {type(e).__name__}")
        time.sleep(45)


def download(model_id, out):
    c = hub.Client()
    from qai_hub import public_rest_api as p
    url = p.get_model_download_url(c.config, model_id).url
    parts = out + ".parts"
    os.makedirs(parts, exist_ok=True)
    r = hub.requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    total = int(r.headers["Content-Range"].split("/")[1])
    r.close()
    ch = 8 << 20
    nch = (total + ch - 1) // ch
    log(f"download {total/1e6:.0f} MB, {nch} chunks")
    q = queue.Queue()
    for i in range(nch):
        q.put(i)
    lock = threading.Lock()
    st = {"done": 0, "bytes": 0}
    t0 = time.time()

    def fetch(i):
        a, b = i * ch, min(i * ch + ch - 1, total - 1)
        want = b - a + 1
        f = f"{parts}/{i:05d}"
        if os.path.exists(f) and os.path.getsize(f) == want:
            return want
        for _ in range(6):
            try:
                rr = hub.requests.get(url, headers={"Range": f"bytes={a}-{b}"},
                                      timeout=(20, 120), stream=True)
                got = 0
                with open(f, "wb") as fh:
                    for c2 in rr.iter_content(262144):
                        fh.write(c2)
                        got += len(c2)
                rr.close()
                if got == want:
                    return want
            except Exception:                        # noqa: BLE001
                pass
            time.sleep(1)
        return 0

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            n = fetch(i)
            with lock:
                st["done"] += 1
                st["bytes"] += n
                if st["done"] % 20 == 0:
                    el = time.time() - t0
                    log(f"{st['done']}/{nch} {st['bytes']/1e6:.0f} MB "
                        f"{st['bytes']/1e6/el:.2f} MB/s")

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if st["done"] != nch:
        log(f"INCOMPLETE {st['done']}/{nch}")
        return False
    with open(out, "wb") as f:
        for i in range(nch):
            with open(f"{parts}/{i:05d}", "rb") as g:
                f.write(g.read())
    log(f"saved {out} {os.path.getsize(out)/1e6:.0f} MB")
    return True


def main():
    cmd = sys.argv[1]
    pins()
    c = hub.Client()
    st = st_read()

    if cmd == "quant":
        z = np.load(f"{HOME}/cal_data15.npz")
        data = {"log": [z["log"][i].astype(np.float32) for i in range(z["log"].shape[0])],
                "valid_frames": [z["vf"][i].astype(np.float32) for i in range(z["vf"].shape[0])]}
        np.savez(f"{HOME}/cal_data15_f32.npz",
                 log=np.stack(data["log"]), vf=np.stack(data["valid_frames"]))
        ds = c.upload_dataset(data, name="fe_cal15_f32")
        log(f"dataset {ds.dataset_id}")
        j = c.submit_quantize_job(c.get_model(SRC_MODEL), ds,
                                  weights_dtype=hub.QuantizeDtype.INT8,
                                  activations_dtype=hub.QuantizeDtype.INT16,
                                  name="fe_bb_w8a16_fp32src")
        st_write(dataset=ds.dataset_id, quantize=j.job_id)
        log(f"QUANTIZE {j.job_id}")

    elif cmd == "qc-poll":
        code = wait(st["quantize"], "quantize")
        if code == "SUCCESS":
            j = c.get_job(st["quantize"])
            st_write(qdq=j.get_target_model().model_id)
            log(f"qdq model {j.get_target_model().model_id}")

    elif cmd == "qc-fetch":
        download(st["qdq"], QDQ)

    elif cmd == "ortcheck":
        import onnxruntime as ort
        sess = ort.InferenceSession(QDQ, providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]
        log(f"inputs {names}")
        logmel = np.fromfile(f"{HOME}/fe_logmel.f32", np.float32).reshape(1, 802, 80)
        vf = np.array([[154.0]], np.float32)
        feeds = {}
        for n in names:
            feeds[n] = vf if "valid" in n else logmel
        out = sess.run(None, feeds)[0].reshape(1024, 200)
        ref = np.load(f"{HOME}/feat_ref_24960.npy").reshape(1024, 200)
        a, b = out[:, :39], ref[:, :39]
        cos = float(np.dot(a.ravel(), b.ravel()) /
                    (np.linalg.norm(a) * np.linalg.norm(b)))
        log("QDQ vs fp32: maxabs %.4f relL2 %.5f cos %.7f" %
            (np.abs(a - b).max(), np.linalg.norm(a - b) / np.linalg.norm(b), cos))

    elif cmd == "compile":
        opts = ("--target_runtime qnn_dlc --truncate_64bit_io "
                "--qairt_version 2.51")
        j = c.submit_compile_job(model=c.get_model(st["qdq"]),
                                 device=hub.Device("Samsung Galaxy S25 (Family)"),
                                 name="fe_qdq_i16", options=opts)
        st_write(compile=j.job_id)
        log(f"COMPILE {j.job_id}")

    elif cmd == "link":
        j = c.get_job(st["compile"])
        lj = c.submit_link_job(models=[j.get_target_model()], device=j.device,
                               name="fe_qdq_i16_link",
                               options="--qairt_version 2.51")
        st_write(link=lj.job_id)
        log(f"LINK {lj.job_id}")

    elif cmd == "fetch":
        download(st["link"], BIN)


if __name__ == "__main__":
    main()
