#!/usr/bin/env python3
"""One-shot upload+compile of the 1-layer probe (step-by-step prints)."""
import json
import os

import qai_hub as hub

HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = ("--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type float16 "
        "--qairt_version 2.51")
print("[1] upload", flush=True)
m = hub.upload_model(os.path.join(HERE, "llm_1l.onnx"))
print("[2] uploaded", m.model_id, flush=True)
j = hub.submit_compile_job(model=m, device=hub.Device("Samsung Galaxy S25 (Family)"),
                           name="llm1l_fp16", options=OPTS)
print("[3] COMPILE", j.job_id, flush=True)
json.dump({"1": {"compile": j.job_id}},
          open(os.path.join(HERE, "layer_jobs.json"), "w"), indent=1)
