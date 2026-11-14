#!/usr/bin/env python3
"""Compile the staticized dualcodec encoder CORE (192900 samples / 200 frames)
to a QNN DLC on qai-hub, then link it into a .bin.

usage: core200_compile.py [compile|link <compile_job_id>]
"""
import os
import shutil
import sys
import time

import qai_hub

OUT = "/data/data/com.termux/files/usr/tmp"
ONNX = OUT + "/encode_core_s200.onnx"
DEVICE = "Samsung Galaxy S25 (Family)"
QAIRT = "2.51"
OPTIONS = (f"--target_runtime qnn_dlc --truncate_64bit_io "
           f"--quantize_full_type float16 --qairt_version {QAIRT}")
SPECS = {"audio": ((1, 1, 192900), "float32"),
         "semantic_repr": ((1, 1024, 200), "float32")}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def compile_(onnx=None, n=None, f=None):
    path = onnx or ONNX
    specs = ({"audio": ((1, 1, n), "float32"),
              "semantic_repr": ((1, 1024, f), "float32")}
             if n and f else SPECS)
    dev = next(d for d in qai_hub.get_devices() if d.name == DEVICE)
    log(f"device {dev.name} {getattr(dev, 'device_id', getattr(dev, 'id', ''))}")
    log(f"uploading {path} ({os.path.getsize(path)/1e6:.1f} MB) {specs}")
    job = qai_hub.submit_compile_job(model=path, device=dev,
                                     input_specs=specs, options=OPTIONS)
    log(f"compile job {job.job_id}")
    with open(OUT + "/core200_job.txt", "w") as f:
        f.write(job.job_id)
    job.wait()
    st = job.get_status()
    log(f"status {st.state}")
    if "SUCCESS" not in str(st.state):
        try:
            job.download_artifacts_for_type(OUT, qai_hub.JobArtifactType.HUB_LOG)
            log("hub log downloaded")
        except Exception as e:      # noqa: BLE001
            log(f"no hub log: {e}")
        return
    got = job.download_target_model(OUT + "/encode_core_s200.dlc")
    if got and os.path.abspath(got) != os.path.abspath(OUT + "/encode_core_s200.dlc"):
        shutil.move(got, OUT + "/encode_core_s200.dlc")
    p = OUT + "/encode_core_s200.dlc"
    log(f"dlc {p} {os.path.getsize(p)/1e6:.1f} MB" if os.path.isfile(p) else "no dlc")


def link(job_id):
    job = qai_hub.get_job(job_id)
    model = job.get_target_model()          # already on the server, no re-upload
    dev = next(d for d in qai_hub.get_devices() if d.name == DEVICE)
    log(f"link from model {getattr(model, 'model_id', '?')}")
    lj = qai_hub.submit_link_job(models=[model], device=dev,
                                 options=f"--qairt_version {QAIRT}")
    log(f"link job {lj.job_id}")
    with open(OUT + "/core200_link_job.txt", "w") as f:
        f.write(lj.job_id)
    lj.wait()
    st = lj.get_status()
    log(f"link status {st.state}")
    if "SUCCESS" not in str(st.state):
        try:
            lj.download_artifacts_for_type(OUT, qai_hub.JobArtifactType.HUB_LOG)
            log("hub log downloaded")
        except Exception as e:      # noqa: BLE001
            log(f"no hub log: {e}")
        return
    got = lj.download_target_model(OUT + "/encode_core_s200.bin")
    if got and os.path.abspath(got) != os.path.abspath(OUT + "/encode_core_s200.bin"):
        shutil.move(got, OUT + "/encode_core_s200.bin")
    p = OUT + "/encode_core_s200.bin"
    log(f"bin {p} {os.path.getsize(p)/1e6:.1f} MB" if os.path.isfile(p) else "no bin")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compile"
    if mode == "link":
        link(sys.argv[2])
    else:
        onnx = sys.argv[2] if len(sys.argv) > 2 else None
        n = int(sys.argv[3]) if len(sys.argv) > 3 else None
        f = int(sys.argv[4]) if len(sys.argv) > 4 else None
        compile_(onnx, n, f)
