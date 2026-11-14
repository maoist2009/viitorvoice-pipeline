#!/usr/bin/env python3
"""Compile+link a shorter-window variant of the encoder CORE, reusing the ONNX
model already uploaded by the first compile job (no re-upload).

usage: core_variant.py <audio_samples> <frames> [source_compile_job_id]
"""
import os
import shutil
import sys
import time

import qai_hub

OUT = "/data/data/com.termux/files/usr/tmp"
DEVICE = "Samsung Galaxy S25 (Family)"
SRC_JOB = sys.argv[3] if len(sys.argv) > 3 else "jpy2e4erg"
N = int(sys.argv[1])
F = int(sys.argv[2])
TAG = f"c{N}x{F}"
OPTIONS = ("--target_runtime qnn_dlc --truncate_64bit_io "
           "--quantize_full_type float16 --qairt_version 2.51")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    src = qai_hub.get_job(SRC_JOB)
    model = src.model
    dev = next(d for d in qai_hub.get_devices() if d.name == DEVICE)
    log(f"source model {getattr(model, 'model_id', '?')} -> {N} samples / {F} frames")
    job = qai_hub.submit_compile_job(
        model=model, device=dev,
        input_specs={"audio": ((1, 1, N), "float32"),
                     "semantic_repr": ((1, 1024, F), "float32")},
        options=OPTIONS)
    log(f"compile {job.job_id}")
    job.wait()
    st = job.get_status()
    log(f"compile {st.state}")
    if "SUCCESS" not in str(st.state):
        try:
            job.download_artifacts_for_type(OUT, qai_hub.JobArtifactType.HUB_LOG)
        except Exception as e:      # noqa: BLE001
            log(f"no log {e}")
        return
    tm = job.get_target_model()
    lj = qai_hub.submit_link_job(models=[tm], device=dev,
                                 options="--qairt_version 2.51")
    log(f"link {lj.job_id}")
    lj.wait()
    st = lj.get_status()
    log(f"link {st.state}")
    if "SUCCESS" not in str(st.state):
        try:
            lj.download_artifacts_for_type(OUT, qai_hub.JobArtifactType.HUB_LOG)
        except Exception as e:      # noqa: BLE001
            log(f"no log {e}")
        return
    dst = f"{OUT}/encode_core_{TAG}.bin"
    got = lj.download_target_model(dst)
    if got and os.path.abspath(got) != os.path.abspath(dst):
        shutil.move(got, dst)
    log(f"bin {dst} {os.path.getsize(dst)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
