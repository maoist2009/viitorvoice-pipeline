#!/usr/bin/env python3
"""Run the 5-way split decoder chain on the phone with qnn-net-run and compare
against the ONNX Runtime reference.

usage: verify_chain.py <bins_dir> <ref_dir> <work_dir>

Bounds files come from the runner (~/dec/calib/):
  ref_sem.npy  (1,1,200)  int32
  ref_aco.npy  (1,11,200) int32
  ref_audio.npy (1,1,191996) float32
"""
import os
import subprocess
import sys
import time

import numpy as np

CH, HALO = 47999, 48
QNN = os.path.expanduser("~/qairt/2.51.40")
CL = f"{QNN}/lib/aarch64-android"
BN = f"{QNN}/bin/aarch64-android"
SK = f"{QNN}/lib/hexagon-v79/unsigned"

bins, refdir, work = sys.argv[1], sys.argv[2], sys.argv[3]
tail_bins = sys.argv[4] if len(sys.argv) > 4 else bins
os.makedirs(work, exist_ok=True)

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = f"/vendor/lib64:{CL}"
env["ADSP_LIBRARY_PATH"] = SK


def load_out(path, elems):
    """qnn-net-run writes the graph's native dtype: float32 (4 B) or uint8 (1 B)."""
    n = os.path.getsize(path)
    if n == 4 * elems:
        return np.fromfile(path, dtype=np.float32)
    if n == elems:
        # p1's quantized output encoding from qnn-context-binary-utility
        return (np.fromfile(path, dtype=np.uint8).astype(np.float32) - 120.0) * 0.016783496364951134
    raise SystemExit(f"unexpected output size {n} for {elems} elems")


def run(bin_name, paths, out_dir, elems=None, bins_dir=None):
    """paths: input .raw files in graph input order (one inference)."""
    os.makedirs(out_dir, exist_ok=True)
    lst = os.path.join(out_dir, "input_list.txt")
    with open(lst, "w") as f:
        f.write(" ".join(paths) + "\n")
    cmd = [
        f"{BN}/qnn-net-run",
        "--backend", f"{CL}/libQnnHtp.so",
        "--retrieve_context", os.path.join(bins_dir or bins, bin_name),
        "--input_list", lst,
        "--output_dir", out_dir,
        "--use_native_input_files",
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        raise SystemExit(f"qnn-net-run failed for {bin_name}")
    print(f"  {bin_name}: {dt * 1000:.0f} ms (incl. load)", flush=True)
    res = os.path.join(out_dir, "Result_0")
    outs = {}
    for fn in sorted(os.listdir(res)):
        if fn.endswith(".raw"):
            path = os.path.join(res, fn)
            outs[fn[:-4]] = np.fromfile(path, dtype=np.float32) if elems is None \
                else load_out(path, elems)
    return outs


def raw(path, arr, dtype):
    arr.astype(dtype).tofile(path)
    return list(arr.shape)


sem = np.load(os.path.join(refdir, "ref_sem.npy"))
aco = np.load(os.path.join(refdir, "ref_aco.npy"))
ref = np.load(os.path.join(refdir, "ref_audio.npy")).reshape(-1)

sem_path = os.path.join(work, "sem.raw")
aco_path = os.path.join(work, "aco.raw")
sem.astype(np.int32).tofile(sem_path)
aco.astype(np.int32).tofile(aco_path)

o = run("p1.bin", [sem_path, aco_path], os.path.join(work, "o1"))
conv = list(o.values())[0].reshape(1, 768, 1600)

p = os.path.join(work, "conv.raw")
conv.astype(np.float32).tofile(p)
o = run("p2.bin", [p], os.path.join(work, "o2"), bins_dir=tail_bins)
c1 = list(o.values())[0].reshape(1, 384, 9600)

p = os.path.join(work, "c1.raw")
c1.astype(np.float32).tofile(p)
o = run("p3.bin", [p], os.path.join(work, "o3"), bins_dir=tail_bins)
c2 = list(o.values())[0].reshape(1, 192, 47999)

p = os.path.join(work, "c2.raw")
c2.astype(np.float32).tofile(p)
o = run("p4.bin", [p], os.path.join(work, "o4"), bins_dir=tail_bins)
c3 = list(o.values())[0].reshape(1, 96, 191996)

# p5 in 4 chunks with HALO of context on each side
total = c3.shape[2]
audio = []
for i in range(4):
    start = i * CH - HALO
    end = start + CH + 2 * HALO
    buf = np.zeros((1, 96, CH + 2 * HALO), dtype=np.float32)
    s0, e0 = max(start, 0), min(end, total)
    buf[:, :, s0 - start:e0 - start] = c3[:, :, s0:e0]
    p = os.path.join(work, f"c3_{i}.raw")
    buf.tofile(p)
    o = run("p5chunk48.bin", [p], os.path.join(work, f"o5_{i}"), bins_dir=tail_bins)
    out = list(o.values())[0].reshape(1, 1, CH + 2 * HALO)
    audio.append(out[:, :, HALO:HALO + CH])
got = np.concatenate(audio, axis=2).reshape(-1)

np.save(os.path.join(work, "got.npy"), got.astype(np.float32))
n = min(got.size, ref.size)
g, r = got[:n], ref[:n]
num = float(np.sum((g - r) ** 2))
den = float(np.sum(r ** 2))
snr = 10 * np.log10(den / num) if num > 0 else float("inf")
corr = float(np.corrcoef(g, r)[0, 1])
print(f"len {n}  ref std {r.std():.4f}  got std {g.std():.4f}")
print(f"SNR {snr:.2f} dB   corr {corr:.4f}   max abs err {float(np.abs(g-r).max()):.4f}")
