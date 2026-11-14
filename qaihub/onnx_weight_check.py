#!/usr/bin/env python3
"""Do the exported ONNX weights match the checkpoint exactly?

A bf16-rounded export would explain a ~1e-3 relative drift that the 28-layer
model then amplifies (the residual stream carries a 24k outlier).
"""
import os

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors import safe_open

ONNX = "/data/data/com.termux/files/home/projects/dreamlite/salvage/dl_vv/llm_backbone.onnx"
CKPT = ("/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/"
        "viitor-voice-nar/local_models/llm/0p6_emotion/model.safetensors")


def bf16_round(x):
    import torch
    return torch.from_numpy(x).bfloat16().float().numpy()


def main():
    m = onnx.load(ONNX)
    inits = {t.name: t for t in m.graph.initializer}
    print("initializers", len(inits))
    # named ones first
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith("llm.")]
        print("ckpt tensors", len(keys))
        for name in ("llm.layers.0.input_layernorm.weight",
                     "llm.layers.0.self_attn.q_proj.weight",
                     "llm.layers.27.mlp.down_proj.weight",
                     "llm.norm.weight"):
            short = name[4:]
            hit = None
            for k in inits:
                if k.endswith(short) or k == name:
                    hit = k
                    break
            ref = f.get_tensor(name).float().numpy()
            if hit is None:
                print(f"  {name}: not found by name")
                continue
            got = numpy_helper.to_array(inits[hit])
            d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
            print(f"  {hit}: shape {got.shape} dtype {got.dtype} "
                  f"max|d| {d.max():.3e} rel {d.max()/max(1e-9, np.abs(ref).max()):.3e} "
                  f"bf16-exact {np.abs(bf16_round(ref)-ref).max():.3e}")
    # biggest anonymous initializer vs q_proj of layer 0
    big = sorted(m.graph.initializer, key=lambda t: -t.ByteSize())[:4]
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        q = f.get_tensor("llm.layers.0.self_attn.q_proj.weight").float().numpy()
        gate = f.get_tensor("llm.layers.0.mlp.gate_proj.weight").float().numpy()
    for t in big:
        arr = numpy_helper.to_array(t)
        for ref, tag in ((q, "q0"), (gate, "gate0")):
            if arr.shape == ref.shape:
                d = np.abs(arr.astype(np.float64) - ref.astype(np.float64))
                print(f"  {t.name}: {arr.shape} matches {tag} max|d| {d.max():.3e} "
                      f"(bf16 noise would be {np.abs(bf16_round(ref)-ref).max():.3e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
