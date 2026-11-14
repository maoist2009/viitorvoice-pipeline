#!/usr/bin/env python3
"""Offline export helpers for the W2V-BERT 2.0 semantic frontend.

Two artifacts are needed by the native (C++) W2VFrontend:

  1. w2v_bert_2_0.onnx
     - Source: HF `dualcodec/w2v-bert-2.0` (PyTorch only).
     - Export with `transformers` + `onnxruntime` on a machine with a GPU.
     - Take hidden_states[output_idx] (penultimate ~= -2).
     - The C++ side does: (feat - mean)/std, transpose(1,2), avg_pool1d(k=2)
       to produce semantic_repr(1,1024,F).

  2. w2vbert2_mean_var_stats_emilia_mean.bin / _std.bin
     - Source: w2vbert2_mean_var_stats_emilia.pt  (dict {mean(1024,), var(1024,)})
     - Converted here to float32 row-major .bin (std = sqrt(var)).
     - RUNNABLE NOW (only needs torch):  python tools/export_w2v_onnx.py --stats

Usage:
    python tools/export_w2v_onnx.py --stats
    python tools/export_w2v_onnx.py --onnx /path/to/w2v-bert-2.0 --out w2v_bert_2_0.onnx
"""

import argparse
import math
import os
import sys

import torch


def export_stats(pt_path: str, out_dir: str) -> None:
    sd = torch.load(pt_path, map_location="cpu")
    mean = sd["mean"].float().reshape(-1)          # (1024,)
    std = torch.sqrt(sd["var"]).float().reshape(-1)  # (1024,)
    assert mean.numel() == 1024 and std.numel() == 1024, (mean.shape, std.shape)
    mean.numpy().tofile(os.path.join(out_dir, "w2vbert2_mean_var_stats_emilia_mean.bin"))
    std.numpy().tofile(os.path.join(out_dir, "w2vbert2_mean_var_stats_emilia_std.bin"))
    print(f"[stats] wrote mean/std ({mean.numel()} f32 each) -> {out_dir}")


def export_onnx(model_dir: str, out_path: str, output_idx: int = -2) -> None:
    from transformers import AutoModel, AutoFeatureExtractor
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.tools import pytorch_export

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()
    fe = AutoFeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)

    class W2VWrapper(torch.nn.Module):
        """Forward returns the selected hidden state only (1, 1024, F)."""
        def __init__(self, m, idx):
            super().__init__()
            self.m = m
            self.idx = idx

        def forward(self, input_values):
            out = self.m(input_values, output_hidden_states=True)
            h = out.hidden_states[self.idx]      # (1, F, 1024)
            return h                             # C++ side does norm+transpose+pool

    dummy = torch.zeros(1, 16000, dtype=torch.float32)
    wrapper = W2VWrapper(model, output_idx)
    wrapper.eval()
    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=["input_values"], output_names=["hidden_states"],
        opset_version=14, do_constant_folding=True,
    )
    print(f"[onnx] wrote {out_path}")

    # Sanity check the exported graph.
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input_values": np.zeros((1, 16000), dtype=np.float32)})
    print(f"[onnx] smoke shape: {got[0].shape}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="convert .pt -> mean/std .bin")
    ap.add_argument("--onnx", metavar="MODEL_DIR", help="export w2v_bert_2_0.onnx from HF dir")
    ap.add_argument("--out", default="w2v_bert_2_0.onnx", help="ONNX output path")
    ap.add_argument("--output-idx", type=int, default=-2, help="hidden_states index to emit")
    ap.add_argument("--stats-pt",
                    default="viitor-voice-nar/local_models/dualcodec/dualcodec_ckpts/w2vbert2_mean_var_stats_emilia.pt")
    ap.add_argument("--out-dir", default="viitor-voice-nar/local_models/dualcodec/dualcodec_ckpts")
    args = ap.parse_args()

    if args.stats:
        export_stats(args.stats_pt, args.out_dir)
    if args.onnx:
        export_onnx(args.onnx, args.out, args.output_idx)
    if not args.stats and not args.onnx:
        print("nothing to do; pass --stats and/or --onnx")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
