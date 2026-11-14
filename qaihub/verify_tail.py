#!/usr/bin/env python3
"""Device-side check of the decoder tail (p2 -> p5 chunked) against the ORT
reference, starting from the `convolution` boundary tensor.

usage: verify_tail.py <bins_dir> <data_dir> <work_dir>

data_dir files: s0_convolution.npy, ref_audio2.npy
"""
import os
import subprocess
import sys

import numpy as np

CH, HALO = 47999, 48
QNN = os.path.expanduser("~/qairt/2.51.40")
CL = f"{QNN}/lib/aarch64-android"
BN = f"{QNN}/bin/aarch64-android"
SK = f"{QNN}/lib/hexagon-v79/unsigned"

bins, data, work = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(work, exist_ok=True)
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = f"/vendor/lib64:{CL}"
env["ADSP_LIBRARY_PATH"] = SK


def run(bin_name, paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lst = os.path.join(out_dir, "input_list.txt")
    with open(lst, "w") as f:
        f.write(" ".join(paths) + "\n")
    cmd = [f"{BN}/qnn-net-run", "--backend", f"{CL}/libQnnHtp.so",
           "--retrieve_context", os.path.join(bins, bin_name),
           "--input_list", lst, "--output_dir", out_dir]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        raise SystemExit(f"qnn-net-run failed for {bin_name}")
    res = os.path.join(out_dir, "Result_0")
    return {fn[:-4]: np.fromfile(os.path.join(res, fn), dtype=np.float32)
            for fn in sorted(os.listdir(res)) if fn.endswith(".raw")}


def stage(name, arr):
    p = os.path.join(work, name + ".raw")
    arr.astype(np.float32).tofile(p)
    return p


def snr(g, r):
    n = min(g.size, r.size)
    g, r = g[:n], r[:n]
    num = float(np.sum((g - r) ** 2))
    den = float(np.sum(r ** 2))
    s = 10 * np.log10(den / num) if num > 0 else float("inf")
    return s, float(np.corrcoef(g, r)[0, 1]), float(np.abs(g - r).max())


x = np.load(os.path.join(data, "s0_convolution.npy"))
o = run("p2.bin", [stage("conv", x)], os.path.join(work, "o2"))
c1 = list(o.values())[0]
print("p2 out", c1.shape, end="  ")

o = run("p3.bin", [stage("c1", c1)], os.path.join(work, "o3"))
c2 = list(o.values())[0]
print("p3 out", c2.shape, end="  ")

o = run("p4.bin", [stage("c2", c2)], os.path.join(work, "o4"))
c3 = list(o.values())[0].reshape(1, 96, 191996)
print("p4 out", c3.shape)

total = c3.shape[2]
parts = []
for i in range(4):
    start = i * CH - HALO
    end = start + CH + 2 * HALO
    buf = np.zeros((1, 96, CH + 2 * HALO), dtype=np.float32)
    s0, e0 = max(start, 0), min(end, total)
    buf[:, :, s0 - start:e0 - start] = c3[:, :, s0:e0]
    o = run("p5chunk48.bin", [stage(f"c3_{i}", buf)], os.path.join(work, f"o5_{i}"))
    out = list(o.values())[0].reshape(1, 1, CH + 2 * HALO)
    parts.append(out[:, :, HALO:HALO + CH])
got = np.concatenate(parts, axis=2).reshape(-1)
ref = np.load(os.path.join(data, "ref_audio2.npy")).reshape(-1)

s, c, m = snr(got, ref)
print(f"TAIL  ref std {ref.std():.4f}  got std {got.std():.4f}")
print(f"TAIL  SNR {s:.2f} dB  corr {c:.4f}  max abs err {m:.4f}")
