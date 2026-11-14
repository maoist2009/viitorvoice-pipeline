#!/usr/bin/env python3
"""Staticize the dualcodec encoder through a qai-hub ONNX-runtime job.

qai-hub rejects the dynamic encoder ONNX when `--quantize_io` is off, and
`--quantize_io` itself is unusable here (its heuristic [0,1] IO range clips the
graph's signed inputs: the 24 kHz waveform and the w2v hidden states).

The documented fix is a compile job that targets ONNX runtime with explicit
input shapes; it returns a staticized .onnx which is then compiled to QNN with
fp32 IO.

usage: encoder_shapefix.py submit | poll | compile | poll-compile | link | poll-link | fetch
"""
import json
import os
import sys

import qai_hub as hub

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "encoder_fix.json")
SRC_JOB = "jgd6872lp"          # original encoder compile (carries model + calibration)
ONNX_OUT = os.path.join(HERE, "encode_static.onnx")


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(d):
    json.dump(d, open(STATE, "w"), indent=1)


def main():
    cmd = sys.argv[1]
    c = hub.Client()
    st = load()

    if cmd == "submit":
        src = c.get_job(SRC_JOB)
        j = c.submit_compile_job(
            model=src.model, device=src.device, name="encoder_staticize",
            options="--target_runtime onnx",
            input_specs={"audio": ((1, 1, 240000), "float32"),
                         "semantic_repr": ((1, 1024, 600), "float32")})
        st["shapefix"] = j.job_id
        save(st)
        print("shapefix", j.job_id, flush=True)

    elif cmd == "poll":
        s = c.get_job(st["shapefix"]).get_status()
        print("shapefix", s.code, s.message[:200], flush=True)

    elif cmd == "fetch-static":
        j = c.get_job(st["shapefix"])
        if j.get_status().code != "SUCCESS":
            raise SystemExit("shapefix not done: " + j.get_status().code)
        p = j.download_target_model(ONNX_OUT)
        print("static onnx ->", p or ONNX_OUT, os.path.getsize(ONNX_OUT), "B", flush=True)

    elif cmd == "compile":
        m = c.upload_model(ONNX_OUT, name="encode_static")
        st["model"] = m.model_id
        save(st)
        j = c.submit_compile_job(
            model=m, device=c.get_job(SRC_JOB).device, name="encoder_fp32io",
            options="--target_runtime qnn_dlc --truncate_64bit_io "
                    "--quantize_full_type int8 --qairt_version 2.51",
            calibration_data=c.get_job(SRC_JOB).calibration_dataset)
        st["compile"] = j.job_id
        save(st)
        print("compile", j.job_id, flush=True)

    elif cmd == "poll-compile":
        s = c.get_job(st["compile"]).get_status()
        print("compile", s.code, s.message[:200], flush=True)

    elif cmd == "link":
        j = c.get_job(st["compile"])
        if j.get_status().code != "SUCCESS":
            raise SystemExit("compile not done: " + j.get_status().code)
        lj = c.submit_link_job(models=j.get_target_model(), device=j.device,
                               name="encoder_fp32io_link",
                               options="--qairt_version 2.51")
        st["link"] = lj.job_id
        save(st)
        print("link", lj.job_id, flush=True)

    elif cmd == "poll-link":
        s = c.get_job(st["link"]).get_status()
        print("link", s.code, s.message[:200], flush=True)

    elif cmd == "fetch":
        j = c.get_job(st["link"])
        if j.get_status().code != "SUCCESS":
            raise SystemExit("link not done: " + j.get_status().code)
        dest = os.path.join(HERE, "fixio", "encoder.bin")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        j.download_target_model(dest)
        print("saved", dest, os.path.getsize(dest), "B", flush=True)

    else:
        raise SystemExit("unknown command")


if __name__ == "__main__":
    main()
