#!/usr/bin/env python3
"""Compile the int8 decoder at SHORT static lengths (100 and 50 frames) to BOTH
tflite and qnn_dlc targets, so we can (a) test the HTP prepare via qtld-net-run,
and (b) produce an app-ready .bin via qairt-dlc-prepare — the proven RVC path.
Halving the sequence halves the waveform-domain conv strips that overflow the
8MB hardware TCM (0xa6b000 at 200 frames)."""
import os, sys, time
import qai_hub as hub

OUT = "/data/data/com.termux/files/home/tmp/dl_vv"
TFL = f"{OUT}/tflite"
DEVICE = "Samsung Galaxy S25 (Family)"
SRC_MODEL = "mq9e08z0m"   # dualcodec_decoder.onnx (already uploaded)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

cl = hub.Client()
devs = hub.get_devices()
s25 = next(d for d in devs if d.name == DEVICE)
src = cl.get_model(SRC_MODEL)

results = []
for ln in (100, 50):
    for tgt in ("tflite", "qnn_dlc"):
        opts = f"--target_runtime {tgt} --truncate_64bit_io --quantize_full_type int8 --quantize_io"
        log(f"--- decoder int8 {tgt} len={ln} ---")
        job = hub.submit_compile_job(
            model=src, device=s25,
            input_specs={"semantic_codes": ((1, 1, ln), "int64"),
                         "acoustic_codes": ((1, 11, ln), "int64")},
            options=opts)
        log(f"  job {job.job_id}")
        try:
            job.wait(timeout=2400)
        except Exception as e:
            log(f"  wait failed {e}")
        st = job.get_status()
        log(f"  status {st.state} | {str(st.message)[:200]}")
        if str(st.state) != "State.SUCCESS":
            continue
        ext = "tflite" if tgt == "tflite" else "dlc"
        dst = (f"{TFL}/decoder_int8_len{ln}.tflite" if tgt == "tflite"
               else f"{OUT}/dlc/decoder_int8_len{ln}.dlc")
        got = job.download_target_model(dst)
        if got:
            log(f"  -> {got} ({os.path.getsize(got)/1e6:.1f} MB)")
            results.append(got)

# zero inputs for each length
import numpy as np
for ln in (100, 50):
    np.array([0]*ln, dtype=np.int32).tofile(f"{TFL}/sem{ln}.raw")
    np.zeros((1, 11, ln), dtype=np.int32).tofile(f"{TFL}/ac{ln}.raw")
    with open(f"{TFL}/input_list_len{ln}.txt", "w") as f:
        f.write(f"# semantic_codes\nsem{ln}.raw\n# acoustic_codes\nac{ln}.raw\n")
log("ALL DONE")
for r in results: log("  artifact " + r)
