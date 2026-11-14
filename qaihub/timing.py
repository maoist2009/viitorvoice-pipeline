#!/usr/bin/env python3
"""Separate context-load time from per-inference execution time.

qnn-net-run reloads the context binary in every process, which hides the real
per-frame cost. Running the same input N times inside one process and comparing
N=2 against N=20 gives exec = (t20-t2)/18 and load = t2 - 2*exec.

usage: timing.py <bins_dir> <work_dir>
"""
import os
import subprocess
import sys
import time

QNN = os.path.expanduser("~/qairt/2.51.40")
CL, BN, SK = (f"{QNN}/lib/aarch64-android", f"{QNN}/bin/aarch64-android",
              f"{QNN}/lib/hexagon-v79/unsigned")
bins, work = sys.argv[1], sys.argv[2]
env = dict(os.environ, LD_LIBRARY_PATH=f"/vendor/lib64:{CL}", ADSP_LIBRARY_PATH=SK)


def timed(bin_name, paths, n):
    lst = os.path.join(work, f"l_{bin_name}_{n}.txt")
    with open(lst, "w") as f:
        f.write(" ".join(paths) + "\n")
    out = os.path.join(work, f"t_{bin_name}_{n}")
    cmd = [f"{BN}/qnn-net-run", "--backend", f"{CL}/libQnnHtp.so",
           "--retrieve_context", os.path.join(bins, bin_name),
           "--input_list", lst, "--output_dir", out,
           "--use_native_input_files", "--num_inferences", str(n)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        print(p.stdout[-800:], p.stderr[-800:])
        raise SystemExit(f"failed: {bin_name}")
    return dt


def measure(bin_name, paths, label):
    t2 = timed(bin_name, paths, 2)
    t20 = timed(bin_name, paths, 20)
    ex = (t20 - t2) / 18
    load = t2 - 2 * ex
    print(f"{label:<12} exec {ex * 1000:7.1f} ms/inf   load {load * 1000:6.0f} ms", flush=True)
    return ex


sem = os.path.join(work, "sem.raw")
aco = os.path.join(work, "aco.raw")
ex1 = measure("p1.bin", [sem, aco], "p1 (fp16)")
ex2 = measure("p2.bin", [os.path.join(work, "conv.raw")], "p2")
ex3 = measure("p3.bin", [os.path.join(work, "c1.raw")], "p3")
ex4 = measure("p4.bin", [os.path.join(work, "c2.raw")], "p4")
ex5 = measure("p5chunk48.bin", [os.path.join(work, "c3_0.raw")], "p5 (1 chunk)")
print(f"\ntotal per 8 s of audio: {(ex1 + ex2 + ex3 + ex4 + 4 * ex5):.2f} s "
      f"({8 / (ex1 + ex2 + ex3 + ex4 + 4 * ex5):.2f}x realtime)")
