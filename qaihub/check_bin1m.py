#!/usr/bin/env python3
"""Check a compiled LLM context binary against the torch reference hidden states.

usage: check_bin1m.py [bin] [ref.npy]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402

W = "/data/data/com.termux/files/usr/tmp/vv_nar"
SEQ = 256
BIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(W, "bin1m/job_jpvmmewmp_linked_bin_mn0xx4glz.bin")
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(W, "ref_h28.npy")


def main():
    emb = np.load(os.path.join(W, "emb256.npy")).astype(np.float32).reshape(1, SEQ, 1024)
    mask = np.zeros((1, 1, SEQ, SEQ), dtype=np.float32)
    ip = os.path.join(W, "chk_in_emb.raw")
    mp = os.path.join(W, "chk_in_mask.raw")
    emb.tofile(ip)
    mask.tofile(mp)
    outs, dt = P.net_run(BIN, [ip, mp], os.path.join(W, "chk_out"),
                         [("output_0", SEQ * 1024 * 4)])
    got = outs[0].reshape(SEQ, 1024).astype(np.float64)
    ref = np.load(REF).astype(np.float64).reshape(SEQ, 1024)
    c = (got * ref).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(ref, axis=-1))
    print(f"{os.path.basename(BIN)}  {dt:.1f} ms")
    print(f"cos mean {c.mean():.6f} min {c.min():.6f}  max|d| {np.abs(got - ref).max():.4f}"
          f"  max|ref| {np.abs(ref).max():.2f}  max|got| {np.abs(got).max():.2f}")
    print("per-16-frame cos:", " ".join(f"{c[i:i+16].mean():.4f}" for i in range(0, SEQ, 16)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
