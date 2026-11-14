#!/usr/bin/env python3
"""Is the compiled fp16 llm backbone really floating point end to end?

RMSNorm makes the backbone mathematically scale-invariant in the input
embedding scale, so feeding emb, emb*1000 and emb*0.001 must give the same
hidden states.  If any int8/uint8 activation quantization sneaks into the DLC,
the large/small inputs saturate and the outputs diverge.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402

W = "/data/data/com.termux/files/usr/tmp/vv_nar"
SEQ, DIM = 256, 1024
FP16 = os.path.join(P.HERE, "fp16/job_jgnq1k3vg_linked_bin_mqy33gjor.bin")


def run(emb, tag):
    emb.astype(np.float32).tofile(os.path.join(W, "emb.raw"))
    attn = np.zeros((1, 1, SEQ, SEQ), dtype=np.float32)
    attn.tofile(os.path.join(W, "attn_full.raw"))
    (hid,), ms = P.net_run(
        FP16, [os.path.join(W, "emb.raw"), os.path.join(W, "attn_full.raw")],
        os.path.join(W, "llm_" + tag), [("output_0.raw", SEQ * DIM * 4)])
    print(f"[{tag}] {ms:.0f}ms std {hid.std():.4f} norm {np.linalg.norm(hid):.2f}",
          flush=True)
    return hid.astype(np.float64)


def cos(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    emb = np.load(os.path.join(W, "emb256.npy")).astype(np.float32)
    print("emb range", emb.min(), emb.max(), flush=True)
    h1 = run(emb, "s1")
    h2 = run(emb * 1000.0, "s1000")
    h3 = run(emb * 0.001, "s0.001")
    print("cos(s1, s1000)  %.5f" % cos(h1, h2))
    print("cos(s1, s0.001) %.5f" % cos(h1, h3))
    np.save(os.path.join(W, "hid_scale1.npy"), h1)
    np.save(os.path.join(W, "hid_scale1000.npy"), h2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
