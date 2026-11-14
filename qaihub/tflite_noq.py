#!/usr/bin/env python3
"""(1) Download the HUB_LOG of the successful farm inference job jgk68qey5 — it may
reveal which backend the farm's runtime used (htp delegate vs cpu).
(2) Submit a NO-quantize tflite compile of the decoder, reusing the already-uploaded
source model (avoids the flaky 268 MB S3 upload). This is the direct analog of the
proven RVC recipe (--target_runtime <x> --truncate_64bit_io, no quantize flags)."""
import os, time
import qai_hub as hub

OUT = "/data/data/com.termux/files/home/tmp/dl_vv/tflite"
DEVICE = "Samsung Galaxy S25 (Family)"
SRC_MODEL = "mq9e08z0m"   # dualcodec_decoder.onnx, already on qai-hub

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

cl = hub.Client()

# ---- (1) farm inference log -------------------------------------------------
log("--- inference job jgk68qey5 HUB_LOG ---")
ij = cl.get_job("jgk68qey5")
try:
    p = ij.download_artifacts_for_type(OUT, hub.JobArtifactType.HUB_LOG)
    log(f"  hub log -> {p}")
except Exception as e:
    log(f"  (no hub log: {e})")
try:
    ij.download_results_to_folder(f"{OUT}/infer_out")
    log("  results downloaded")
except Exception as e:
    log(f"  (no results: {e})")

# ---- (2) no-quantize tflite compile -----------------------------------------
log("--- submit decoder tflite WITHOUT quantize flags ---")
devs = hub.get_devices()
s25 = next(d for d in devs if d.name == DEVICE)
src = cl.get_model(SRC_MODEL)
log(f"  source model {src.model_id} name={getattr(src,'name',None)}")
job = hub.submit_compile_job(
    model=src, device=s25,
    input_specs={"semantic_codes": ((1, 1, 200), "int64"),
                 "acoustic_codes": ((1, 11, 200), "int64")},
    options="--target_runtime tflite --truncate_64bit_io")
log(f"  compile job {job.job_id}")
with open(f"{OUT}/decoder_tflite_noq_job.txt", "w") as f:
    f.write(job.job_id + "\n")
job.wait(timeout=1800)
st = job.get_status()
log(f"  status {st.state} | {str(st.message)[:300]}")
if str(st.state) == "State.SUCCESS":
    got = job.download_target_model(f"{OUT}/decoder_noq.tflite")
    log(f"  tflite -> {got} ({os.path.getsize(got)/1e6:.1f} MB)" if got else "  NO MODEL")
