#!/usr/bin/env python3
"""Overflow hypothesis: the layer-2+ residual stream carries a ~24k outlier, sox^2 in fp16 overflows (65504) and RMSNorm breaks.

If that is the mechanism, shrinking the input scale (RMSNorm makes the
per-layer math mostly scale-free) must bring the DLC and torch back together.
"""
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402

W = "/data/data/com.termux/files/usr/tmp/vv_nar"
SEQ, DIM = 256, 1024
BIN = os.path.join(P.HERE, "fp16/job_jgnq1k3vg_linked_bin_mqy33gjor.bin")


def torch_ref(emb, scale):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                      num_hidden_layers=28, num_attention_heads=16,
                      num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
                      rope_theta=1000000, max_position_embeddings=40960,
                      tie_word_embeddings=False, attention_bias=False, hidden_act="silu")
    cfg._attn_implementation = "eager"
    m = Qwen3Model(cfg).to(torch.float32)
    sd = {}
    with safe_open(os.path.join(P.MODEL, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("llm.") and "embed_tokens" not in k:
                sd[k[4:]] = f.get_tensor(k).float()
    m.load_state_dict(sd, strict=False)
    m.eval()
    with torch.no_grad():
        out = m(inputs_embeds=torch.from_numpy((emb * scale).astype(np.float32)),
                attention_mask=torch.zeros(1, 1, SEQ, SEQ,
                                           dtype=torch.bool)).last_hidden_state
    return out[0].numpy().astype(np.float64)


def graph(emb, tag):
    emb.astype(np.float32).tofile(os.path.join(W, "emb.raw"))
    np.zeros((1, 1, SEQ, SEQ), dtype=np.float32).tofile(os.path.join(W, "attn_full.raw"))
    (hid,), ms = P.net_run(BIN, [os.path.join(W, "emb.raw"),
                                 os.path.join(W, "attn_full.raw")],
                           os.path.join(W, "llm_" + tag),
                           [("output_0.raw", SEQ * DIM * 4)])
    return hid.astype(np.float64).reshape(SEQ, DIM), ms


def main():
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    emb = np.load(os.path.join(W, "emb256.npy")).astype(np.float32)
    for scale in (1.0, 0.01, 0.001):
        t = torch_ref(emb, scale)
        g, ms = graph(emb * scale, "sc%d" % int(scale * 1000))
        c = (g * t).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(t, axis=-1))
        print(f"scale {scale:<6} ({ms:.0f} ms): cos mean {c.mean():.4f} min {c.min():.4f} "
              f"pos0 {c[0]:.4f} | torch max|h| {np.abs(t).max():.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
