#!/usr/bin/env python3
"""Does a big activation break RMSNorm inside the DLC?

The real backbone produces a ~24k outlier from layer 2 on.  x^2 = 5.8e8 does
not fit in fp16, so a naive Pow->ReduceMean->Sqrt->Reciprocal chain saturates
and the normalisation collapses to 0.  The tiny compiled graph is numerically
exact, so feed it the same kind of outlier and see whether it survives.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402
from probe_tiny import ATTN, EMB, HID, SEQ, STATE  # noqa: E402

W = "/data/data/com.termux/files/usr/tmp/probe_tiny"


def main():
    import onnxruntime as ort
    st = json.load(open(STATE))
    binp = os.path.join(W, "bin", sorted(os.listdir(os.path.join(W, "bin")))[0])
    sess = ort.InferenceSession(os.path.join(W, "tiny.onnx"),
                                providers=["CPUExecutionProvider"])
    base = np.fromfile(EMB, dtype=np.float32).reshape(1, SEQ, HID)
    attn = np.zeros((1, 1, SEQ, SEQ), dtype=np.float32)
    for spike in (0.0, 200.0, 1000.0, 8000.0, 24000.0, 60000.0):
        emb = base.copy()
        if spike:
            emb[0, :, 5] += spike
        ref = sess.run(None, {"inputs_embeds": emb, "attention_mask": attn})[0].reshape(-1)
        emb.astype(np.float32).tofile(os.path.join(W, "emb.raw"))
        attn.tofile(os.path.join(W, "attn.raw"))
        (hid,), ms = P.net_run(binp, [os.path.join(W, "emb.raw"),
                                      os.path.join(W, "attn.raw")],
                               os.path.join(W, "run_spike"),
                               [("output_0.raw", SEQ * HID * 4)])
        got = hid.reshape(-1).astype(np.float64)
        r = ref.astype(np.float64)
        c = float(got @ r / np.linalg.norm(got) / np.linalg.norm(r))
        print(f"spike {spike:>8.0f}: cos {c:.6f}  max|d| {np.abs(got - r).max():.4f}  "
              f"inf {np.isinf(got).any()}  norm ratio "
              f"{np.linalg.norm(got) / np.linalg.norm(r):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
