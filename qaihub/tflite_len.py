#!/usr/bin/env python3
"""Compile the decoder at a SHORTER static sequence so the waveform-domain conv
activations fit in the 8MB hardware TCM. At 200 frames the offending conv needs
0xa6b000 (~10.9MB); at 100 frames the same op should need ~half.
Reuses the already-uploaded fp32 source model (no re-upload)."""
import os, time
import qai_hub as hub

OUT = "/data/data/com.termux/files/home/tmp/dl_vv/tflite"
DEVICE = "Samsung Galaxy S25 (Family)"
SRC_MODEL = "mq9e08z0m"   # dualcodec_decoder.onnx

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

LEN = 100
cl = hub.Client()
devs = hub.get_devices()
s25 = next(d for d in devs if d.name == DEVICE)
src = cl.get_model(SRC_MODEL)
job = hub.submit_compile_job(
    model=src, device=s25,
    input_specs={"semantic_codes": ((1, 1, LEN), "int64"),
                 "acoustic_codes": ((1, 11, LEN), "int64")},
    options="--target_runtime tflite --truncate_64bit_io")
log(f"compile job {job.job_id} at len={LEN}")
with open(f"{OUT}/decoder_tflite_len{LEN}_job.txt", "w") as f:
    f.write(job.job_id + "\n")
job.wait(timeout=1800)
st = job.get_status()
log(f"status {st.state} | {str(st.message)[:300]}")
if str(st.state) == "State.SUCCESS":
    got = job.download_target_model(f"{OUT}/decoder_len{LEN}.tflite")
    log(f"-> {got} ({os.path.getsize(got)/1e6:.1f} MB)" if got else "NO MODEL")

# also write the matching zero inputs for the local prepare test
import struct, numpy as np
np.array([0]*LEN, dtype=np.int32).tofile(f"{OUT}/sem{LEN}.raw")
np.zeros((1, 11, LEN), dtype=np.int32).tofile(f"{OUT}/ac{LEN}.raw")
with open(f"{OUT}/input_list_len{LEN}.txt", "w") as f:
    f.write(f"# semantic_codes\nsem{LEN}.raw\n# acoustic_codes\nac{LEN}.raw\n")
log("inputs written")
