#!/usr/bin/env python3
"""Compile ViiTorVoice dualcodec graphs (fp32 ONNX) to QNN DLC on qai-hub cloud,
then to .bin locally with qairt-dlc-prepare. Mirrors ~/tmp/compile_weihua.py.
Usage: python3 compile_vv.py [decoder|encoder|w2v] [static_len]"""
import os, subprocess, sys, time, shutil
import onnx, qai_hub

SRC = "/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/viitor-voice-nar/local_models"
OUT = "/data/data/com.termux/files/home/tmp/dl_vv"
DEVICE = "Samsung Galaxy S25 (Family)"
QAIRT = "/data/user/0/com.termux/files/usr/lib/qairt"
HTP_BACKEND = QAIRT + "/lib/aarch64-android/libQairtHtp.so"
OPTIONS = "--target_runtime qnn_dlc --truncate_64bit_io"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def run(name, onnx_path, input_specs, static_len=None):
    if not os.path.isfile(onnx_path): log(f"[SKIP] {name}: no {onnx_path}"); return
    log(f"--- {name} ({os.path.getsize(onnx_path)/1e6:.1f} MB) ---")
    devs = qai_hub.get_devices()
    s25 = next(d for d in devs if d.name == DEVICE)
    job = qai_hub.submit_compile_job(model=onnx_path, device=s25,
                                     input_specs=input_specs, options=OPTIONS)
    log(f"  job {job.job_id}")
    job.wait()
    st = job.get_status(); log(f"  status {st.state}")
    if str(st.state) != "State.SUCCESS":
        try: job.download_artifacts_for_type(OUT, qai_hub.JobArtifactType.HUB_LOG)
        except Exception as e: log(f"  (no log {e})")
        log(f"  FAILED"); return
    dlc = f"{OUT}/{name}.dlc"
    got = job.download_target_model(dlc)
    if got and os.path.isfile(got) and os.path.abspath(got) != os.path.abspath(dlc):
        shutil.move(got, dlc)
    if not os.path.isfile(dlc): log("  FAILED no DLC"); return
    log(f"  dlc {dlc} ({os.path.getsize(dlc)/1e6:.1f} MB)")
    cmd = [f"{QAIRT}/bin/aarch64-android/qairt-dlc-prepare",
           f"--input_dlc={dlc}", f"--backend={HTP_BACKEND}",
           f"--binary_file={OUT}/{name}.bin", f"--output_dir={OUT}"]
    env = dict(os.environ)
    env["ADSP_LIBRARY_PATH"] = f"{QAIRT}/lib/hexagon-v79/unsigned"
    env["LD_LIBRARY_PATH"] = "/vendor/lib64:" + f"{QAIRT}/lib/aarch64-android"
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        log(f"  dlc-prepare FAILED rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"); return
    log(f"  bin {OUT}/{name}.bin ({os.path.getsize(OUT + '/' + name + '.bin')/1e6:.1f} MB)")
    with open(f"{OUT}/{name}_job_id.txt", "w") as f: f.write(job.job_id)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "decoder"
    if which == "decoder":
        run("dualcodec_decoder", f"{SRC}/dualcodec/dualcodec_ckpts/dualcodec_decoder.onnx",
            {"semantic_codes": ((1, 1, 200), "int64"),
             "acoustic_codes": ((1, 11, 200), "int64")})
    elif which == "encoder":
        run("dualcodec_encode_core", f"{SRC}/dualcodec/dualcodec_ckpts/dualcodec_encode_core_30s.onnx",
            {"audio": ((1, 1, 240000), "float32"),
             "semantic_repr": ((1, 1024, 600), "float32")})
