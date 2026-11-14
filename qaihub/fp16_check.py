#!/usr/bin/env python3
"""Is the 0.6B backbone safe in fp16 at all?

Runs the real weights in fp32 with per-layer activation stats, then again in
fp16, and reports where fp16 blows up (inf / clipped) compared to fp32.
"""
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402

W = "/data/data/com.termux/files/usr/tmp/vv_nar"
SEQ = 256


def build(dtype):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                      num_hidden_layers=28, num_attention_heads=16,
                      num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
                      rope_theta=1000000, max_position_embeddings=40960,
                      tie_word_embeddings=False, attention_bias=False,
                      hidden_act="silu")
    cfg._attn_implementation = "eager"
    m = Qwen3Model(cfg).to(dtype)
    sd = {}
    with safe_open(os.path.join(P.MODEL, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("llm.") and "embed_tokens" not in k:
                sd[k[4:]] = f.get_tensor(k).to(dtype)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def main():
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    emb = torch.from_numpy(np.load(os.path.join(W, "emb256.npy")).astype(np.float32))

    stats = {}
    m32 = build(torch.float32)
    hooks = []
    for li, layer in enumerate(m32.layers):
        def mk(li):
            def h(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                stats[li] = float(hs.abs().max())
            return h
        hooks.append(layer.register_forward_hook(mk(li)))
    o32 = m32(inputs_embeds=emb, attention_mask=torch.zeros(1, 1, SEQ, SEQ,
                                                            dtype=torch.bool)).last_hidden_state
    for h in hooks:
        h.remove()
    print("fp32 per-layer max|h|:", " ".join(f"{stats[i]:.0f}" for i in sorted(stats)),
          flush=True)
    print("fp32 global max|h| %.1f" % float(o32.abs().max()), flush=True)
    del m32
    import gc
    gc.collect()

    m16 = build(torch.float16)
    o16 = m16(inputs_embeds=emb.half(), attention_mask=torch.zeros(1, 1, SEQ, SEQ,
                                                                 dtype=torch.bool)).last_hidden_state
    o16 = o16.float()
    print("fp16 has inf:", bool(torch.isinf(o16).any()), "nan:",
          bool(torch.isnan(o16).any()), "max|h| %.1f" % float(o16.abs().max()), flush=True)
    a = o16[0].numpy().astype(np.float64)
    b = o32[0].numpy().astype(np.float64)
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    print("torch fp16 vs fp32: cos mean %.5f min %.5f pos0 %.4f"
          % (c.mean(), c.min(), c[0]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
