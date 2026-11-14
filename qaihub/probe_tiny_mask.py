#!/usr/bin/env python3
"""Tiny probe with a *causal* additive mask: does the DLC honour the mask?

Reuses the compiled tiny .bin from probe_tiny.py and the exported ONNX as the
reference (ORT), so both sides see the identical additive mask tensor.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402
from probe_tiny import ATTN, EMB, SEQ, HID, STATE  # noqa: E402

W = "/data/data/com.termux/files/usr/tmp/probe_tiny"
NEG = -30000.0


def main():
    import onnxruntime as ort
    st = json.load(open(STATE))
    binp = os.path.join(W, "bin", sorted(os.listdir(os.path.join(W, "bin")))[0])
    emb = np.fromfile(EMB, dtype=np.float32).reshape(1, SEQ, HID)
    i = np.arange(SEQ)
    tril = np.where(i[None, :] > i[:, None], NEG, 0.0).astype(np.float32)[None, None]
    full = np.zeros((1, 1, SEQ, SEQ), dtype=np.float32)
    sess = ort.InferenceSession(os.path.join(W, "tiny.onnx"),
                                providers=["CPUExecutionProvider"])
    for name, mask in (("full", full), ("tril", tril)):
        ref = sess.run(None, {"inputs_embeds": emb, "attention_mask": mask})[0].reshape(-1)
        emb.tofile(os.path.join(W, "emb.raw"))
        mask.tofile(os.path.join(W, "attn.raw"))
        (hid,), ms = P.net_run(binp, [os.path.join(W, "emb.raw"),
                                      os.path.join(W, "attn.raw")],
                               os.path.join(W, "run_" + name),
                               [("output_0.raw", SEQ * HID * 4)])
        got = hid.reshape(-1).astype(np.float64)
        ref = ref.astype(np.float64)
        c = float(got @ ref / np.linalg.norm(got) / np.linalg.norm(ref))
        per = got.reshape(SEQ, HID), ref.reshape(SEQ, HID)
        cn = (per[0] * per[1]).sum(-1) / (np.linalg.norm(per[0], axis=-1)
                                          * np.linalg.norm(per[1], axis=-1))
        print(f"[{name}] {ms:.0f} ms cos {c:.6f} max|d| {np.abs(got-ref).max():.5f} "
              f"per-token cos min {cn.min():.5f} first4 {np.round(cn[:4],4)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
