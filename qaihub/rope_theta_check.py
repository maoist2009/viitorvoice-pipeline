#!/usr/bin/env python3
"""Is the ONNX/torch mismatch just a RoPE base (rope_theta) difference?

The exporter builds its config from config.json's `llm_config`, which stores
RoPE as {"rope_parameters": {"rope_theta": 1e6, "rope_type": "default"}}.
transformers 4.57 `_compute_default_rope_parameters` reads `config.rope_theta`
(the legacy attribute), which stays at its **default 10000** because the dict
goes into `rope_parameters`.  Every hand-built reference config (torch_generate,
onnx_layer_check) instead passes rope_theta=1e6 explicitly.

So run the *same* torch model twice: theta=10000 (what the ONNX graph bakes)
vs theta=1e6 (what the torch reference used) and compare hidden states.
"""
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

M = os.path.expanduser("~/vvnar/llm/0p6_emotion")
EMB = os.path.expanduser("~/work/emb256.npy")
SEQ = 256


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    cfg_json = json.load(open(os.path.join(M, "config.json")))
    lc = dict(cfg_json["llm_config"])
    mt = lc.pop("model_type")
    lc.pop("architectures", None)
    cfg = AutoConfig.for_model(mt, **lc)
    cfg._attn_implementation = "eager"
    print("config rope_theta =", cfg.rope_theta,
          "rope_parameters =", getattr(cfg, "rope_parameters", None), flush=True)

    llm = AutoModel.from_config(cfg, torch_dtype=torch.float32)
    state = {}
    with safe_open(os.path.join(M, "model.safetensors"), framework="pt") as h:
        for k in h.keys():
            if k.startswith("llm."):
                state[k[4:]] = h.get_tensor(k)
    missing, unexpected = llm.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "rotary_emb.inv_freq"]
    print("missing", missing[:5], "unexpected", unexpected[:5], flush=True)
    llm.eval()

    emb = torch.from_numpy(np.load(EMB).astype(np.float32)).reshape(1, SEQ, 1024)
    mask = torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bool)

    def run(tag):
        with torch.no_grad():
            out = llm(inputs_embeds=emb, attention_mask=mask, output_hidden_states=True)
        hs = [h[0].numpy().astype(np.float64) for h in out.hidden_states]
        print(f"[{tag}] last layer max|h| {np.abs(hs[-1]).max():.2f}", flush=True)
        return hs

    hs10k = run("theta=10000")
    for base in (1000000.0,):
        cfg.rope_theta = base
        cfg.rope_parameters = {"rope_theta": base, "rope_type": "default"}
        llm.rotary_emb = Qwen3RotaryEmbedding(cfg)
        print("now rope_theta =", llm.config.rope_theta, flush=True)
        hs1m = run(f"theta={base:.0f}")
        for li in (0, 1, 2, 3, 7, 15, 27, 28):
            a, b = hs10k[li], hs1m[li]
            c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
            print(f"  theta10k vs theta1m after layer {li:2d}: cos mean {c.mean():.6f} "
                  f"min {c.min():.6f}  max|d| {np.abs(a - b).max():.4f}", flush=True)
        np.savez_compressed(os.path.expanduser("~/work/hs_theta1m.npz"),
                            **{f"h{i}": h for i, h in enumerate(hs1m)})
    np.savez_compressed(os.path.expanduser("~/work/hs_theta10k.npz"),
                        **{f"h{i}": h for i, h in enumerate(hs10k)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
