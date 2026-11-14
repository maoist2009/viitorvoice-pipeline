#!/usr/bin/env python3
"""Exec-only timing for the non-decoder pipeline graphs.

Inputs are all-zero buffers of the graph's declared shape/dtype (we only want
the cost of one graphExecute, not the numerics). exec time is derived from
--num_inferences 2 vs 20 so the context load drops out.

usage: timing_pipeline.py <dlc_dir> <work_dir>
"""
import os
import subprocess
import sys
import time

import numpy as np

QNN = os.path.expanduser("~/qairt/2.51.40")
CL, BN, SK = (f"{QNN}/lib/aarch64-android", f"{QNN}/bin/aarch64-android",
              f"{QNN}/lib/hexagon-v79/unsigned")
dlc, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
env = dict(os.environ, LD_LIBRARY_PATH=f"/vendor/lib64:{CL}", ADSP_LIBRARY_PATH=SK)

# name -> [(tensor, shape, dtype)]
GRAPHS = {
    "w2v_250.bin": [("input_values", (1, 16000), np.float32)],
    "encoder_245.bin": [("semantic_repr", (1, 1024, 600), np.uint8),
                        ("audio", (1, 1, 240000), np.uint8)],
    "llm_245.bin": [("inputs_embeds", (1, 256, 1024), np.uint8),
                    ("attention_mask", (1, 1, 256, 256), np.uint8)],
    "unet_245.bin": [("timestep", (1,), np.uint8),
                     ("text_embeds", (1, 512), np.uint8),
                     ("encoder_hidden_states", (1, 512, 2048), np.uint8),
                     ("sample", (1, 4, 128, 128), np.uint8)],
}


def timed(bin_name, paths, n):
    lst = os.path.join(work, f"l_{bin_name}_{n}.txt")
    with open(lst, "w") as f:
        f.write(" ".join(paths) + "\n")
    out = os.path.join(work, f"t_{bin_name}_{n}")
    cmd = [f"{BN}/qnn-net-run", "--backend", f"{CL}/libQnnHtp.so",
           "--retrieve_context", os.path.join(dlc, bin_name),
           "--input_list", lst, "--output_dir", out,
           "--use_native_input_files", "--num_inferences", str(n)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        print(f"  FAILED {bin_name}: {p.stdout[-400:]}{p.stderr[-400:]}")
        return None
    return dt


for bin_name, spec in GRAPHS.items():
    if not os.path.exists(os.path.join(dlc, bin_name)):
        print(f"{bin_name:<18} missing")
        continue
    paths = []
    for name, shape, dt in spec:
        p = os.path.join(work, f"{bin_name}.{name}.raw")
        np.zeros(shape, dtype=dt).tofile(p)
        paths.append(p)
    t2 = timed(bin_name, paths, 2)
    t20 = timed(bin_name, paths, 20)
    if t2 is None or t20 is None:
        continue
    ex = (t20 - t2) / 18
    print(f"{bin_name:<18} exec {ex * 1000:8.1f} ms/inf   load {(t2 - 2 * ex) * 1000:6.0f} ms",
          flush=True)
