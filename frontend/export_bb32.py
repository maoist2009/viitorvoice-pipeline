#!/usr/bin/env python3
"""Rebuild the fp32 w2v backbone ONNX (log-mel in, semantic_repr out).

The graph the phone runs is not the stock export: the mel frontend lives in C++
and this graph takes its output (`log[802,80]`) plus `valid_frames[1]`,
normalises each mel bin over the valid frames, runs w2v-bert-2.0, takes
hidden_states[-2], applies the emilia stats, transposes and 2x-pools.

It must match the existing fp32 reference bit-for-bit enough for quantisation,
so the last step compares against `feat_ref_24960.npy` produced by the current
graph.

usage: export_bb32.py <model_dir> <out_prefix>
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

MD, OUT = sys.argv[1], sys.argv[2]
ONNX = OUT + ".onnx"
DATA = ONNX + ".data"
T = 802
TILE = "/data/data/com.termux/files/home/tmp/fe_npu"
TILE_R = os.environ.get("REFDIR", ".")


class Wrap(torch.nn.Module):
    def __init__(self, m, idx, nmean, nstd):
        super().__init__()
        self.m = m
        self.idx = idx
        self.register_buffer("nmean", nmean)
        self.register_buffer("nstd", nstd)

    def forward(self, log, valid_frames):
        # per-bin normalisation over the valid mel frames (ddof=1, eps in sqrt)
        nf = log.shape[-2]
        idx = torch.arange(nf, dtype=log.dtype)
        mask = (idx < valid_frames.reshape(())).to(log.dtype)
        n = valid_frames.reshape(())
        mean = (log * mask[:, None]).sum(0) / n
        var = (((log - mean[None, :]) ** 2) * mask[:, None]).sum(0) / (n - 1.0)
        x = (log - mean[None, :]) / torch.sqrt(var + 1e-7)[None, :]

        # the model wants 160-dim stacked frames: pairs share one position
        x = x.reshape(1, nf // 2, 160)
        idx2 = torch.arange(x.shape[1], dtype=log.dtype) * 2.0
        m2 = (idx2 < valid_frames.reshape(())).long().unsqueeze(0)

        out = self.m(input_features=x, attention_mask=m2,
                     output_hidden_states=True)
        h = out.hidden_states[self.idx]
        h = (h - self.nmean) / self.nstd
        return F.avg_pool1d(h.transpose(1, 2), 2, 2)


def main():
    st = np.load(f"{TILE_R}/emilia_stats.npz")
    mean = torch.tensor(st["mean"]).float()
    std = torch.tensor(st["std"]).float()
    model = AutoModel.from_pretrained(MD, trust_remote_code=True).eval()
    n_layers = len(model.encoder.layers)
    idx = -2
    print("layers", n_layers, "hidden idx", idx, flush=True)
    wrap = Wrap(model, idx, mean, std).eval()

    log = torch.from_numpy(np.fromfile(f"{TILE_R}/fe_logmel.f32", np.float32)
                           .reshape(1, T, 80))
    vf = torch.tensor([[154.0]])
    print("inputs", tuple(log.shape), tuple(vf.shape), flush=True)

    with torch.no_grad():
        ref = wrap(log, vf).numpy()
    print("wrap out", ref.shape, float(np.abs(ref).max()), flush=True)

    # dynamo exporter: external data is automatic once the proto passes 2 GiB
    torch.onnx.export(
        wrap, (log[0], vf[0]), ONNX,
        input_names=["log", "valid_frames"],
        output_names=["semantic_repr"],
        opset_version=20, do_constant_folding=True, dynamo=True,
    )
    print("exported", ONNX, os.path.getsize(ONNX),
          os.path.getsize(DATA) if os.path.exists(DATA) else 0, flush=True)

    import onnxruntime as ort
    s = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    got = s.run(None, {"log": log.numpy()[0], "valid_frames": vf.numpy()[0]})[0]
    refv = np.load(f"{TILE_R}/feat_ref_24960.npy").reshape(1024, 200)
    a, b = got[0][:, :39], refv[:, :39]
    cos = float(np.dot(a.ravel(), b.ravel()) /
                (np.linalg.norm(a) * np.linalg.norm(b)))
    print("ORT vs ref: cos %.8f relL2 %.6f maxabs %.4f" %
          (cos, np.linalg.norm(a - b) / np.linalg.norm(b), np.abs(a - b).max()),
          flush=True)

    # the quantiser rejects IO names that also sit in value_info
    import onnx
    m = onnx.load(ONNX, load_external_data=False)
    io = {t.name for t in m.graph.input} | {t.name for t in m.graph.output}
    keep = [v for v in m.graph.value_info if v.name not in io]
    dropped = len(m.graph.value_info) - len(keep)
    del m.graph.value_info[:]
    m.graph.value_info.extend(keep)
    onnx.save(m, ONNX)
    print("value_info dropped", dropped, flush=True)


if __name__ == "__main__":
    main()
