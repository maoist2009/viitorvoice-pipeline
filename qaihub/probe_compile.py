#!/usr/bin/env python3
"""Compile-only probe: does QAIRT accept this graph (e.g. DFT) as qnn_dlc?

usage: probe_compile.py <onnx> <tag>
"""
import sys
import time

import onnx
import qai_hub

ONNX, TAG = sys.argv[1], sys.argv[2]
DEVICE = "Samsung Galaxy S25 (Family)"
OPTIONS = ("--target_runtime qnn_dlc --truncate_64bit_io "
           "--quantize_full_type float16 --qairt_version 2.51")
DT = {1: "float32", 10: "float16", 7: "int64", 6: "int32", 9: "bool"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


m = onnx.load(ONNX, load_external_data=False)
spec = {}
for t in m.graph.input:
    tt = t.type.tensor_type
    spec[t.name] = (tuple(d.dimension[0].dim_value if False else d.dim_value for d in tt.shape.dim),
                    DT.get(tt.elem_type, "float32"))
log(f"inputs {spec}")
dev = next(d for d in qai_hub.get_devices() if d.name == DEVICE)
job = qai_hub.submit_compile_job(model=ONNX, device=dev, input_specs=spec, options=OPTIONS)
log(f"job {job.job_id}")
while True:
    s = qai_hub.get_job(job.job_id).get_status()
    st = str(s.state)
    if st in ("State.SUCCESS", "State.FAILED", "State.CANCELLED"):
        log(f"{TAG} -> {st}  {getattr(s, 'message', '')[:400]}")
        if "SUCCESS" in st:
            tm = qai_hub.get_job(job.job_id).get_target_model()
            log(f"target model {tm.model_type} {tm.name}")
        break
    log(f"{TAG} {st}")
    time.sleep(45)
