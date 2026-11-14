#!/usr/bin/env python3
"""Is the additive attention mask applied transposed inside the DLC?

Feed the graph an *upper* triangular mask (position i may only see j >= i).
If the DLC transposes the mask internally, the graph then behaves like a
normal causal model, so its output must match torch fed with a plain tril
mask.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402

W = "/data/data/com.termux/files/usr/tmp/vv_nar"
SEQ, DIM = 256, 1024
FP16 = os.path.join(P.HERE, "fp16/job_jgnq1k3vg_linked_bin_mqy33gjor.bin")
NEG = -30000.0


def graph(emb, add_mask, tag):
    emb.astype(np.float32).tofile(os.path.join(W, "emb.raw"))
    add_mask.astype(np.float32).tofile(os.path.join(W, "attn_dp.raw"))
    (hid,), ms = P.net_run(FP16, [os.path.join(W, "emb.raw"),
                                  os.path.join(W, "attn_dp.raw")],
                           os.path.join(W, "llm_dp_" + tag),
                           [("output_0.raw", SEQ * DIM * 4)])
    print(f"[graph {tag}] {ms:.0f} ms", flush=True)
    return hid.astype(np.float64).reshape(SEQ, DIM)


def cos_rows(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def main():
    emb = np.load(os.path.join(W, "emb256.npy")).astype(np.float32)
    t_tril = np.load(os.path.join(W, "torch_tril.npy")).reshape(SEQ, DIM)
    i = np.arange(SEQ)
    add_tril = np.where(i[None, :] > i[:, None], NEG, 0.0).astype(np.float32)[None, None]
    add_triu = np.where(i[None, :] < i[:, None], NEG, 0.0).astype(np.float32)[None, None]

    for tag, m, ref, name in (
            ("triu", add_triu, t_tril, "graph(triu) vs torch(tril)"),
            ("tril", add_tril, t_tril, "graph(tril) vs torch(tril)")):
        g = graph(emb, m, tag)
        c = cos_rows(g, ref)
        print(f"{name}: cos mean {c.mean():.4f} pos0 {c[0]:.3f} pos7 {c[7]:.3f} "
              f"pos255 {c[255]:.3f}")
        print("   blocks of 32:", " ".join(f"{c[k:k+32].mean():.3f}"
                                           for k in range(0, 256, 32)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
