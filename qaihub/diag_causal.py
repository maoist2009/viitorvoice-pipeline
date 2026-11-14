#!/usr/bin/env python3
"""Is the compiled backbone *causal*?

The w8a16 build ignored attention_mask entirely (bit-identical output for every
mask we tried) so "full attention" was the only reading.  The fp16 build does
honour the mask, which means the baked-in default may well be causal.  Compare
four combinations: graph/torch x full/causal.
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402

W = "/data/data/com.termux/files/usr/tmp/vv_nar"
SEQ, DIM = 256, 1024
FP16 = os.path.join(P.HERE, "fp16/job_jgnq1k3vg_linked_bin_mqy33gjor.bin")


def graph(emb, attn, tag):
    emb.astype(np.float32).tofile(os.path.join(W, "emb.raw"))
    attn.astype(np.float32).tofile(os.path.join(W, "attn_%s.raw" % tag))
    (hid,), ms = P.net_run(FP16, [os.path.join(W, "emb.raw"),
                                  os.path.join(W, "attn_%s.raw" % tag)],
                           os.path.join(W, "llm_" + tag),
                           [("output_0.raw", SEQ * DIM * 4)])
    print(f"[graph {tag}] {ms:.0f} ms std {hid.std():.4f}", flush=True)
    return hid.astype(np.float64)


def cos(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    emb = np.load(os.path.join(W, "emb256.npy")).astype(np.float32)
    i = np.arange(SEQ)
    full = np.zeros((1, 1, SEQ, SEQ), dtype=np.float32)
    causal = np.where(i[None, :] > i[:, None], -30000.0, 0.0).astype(np.float32)
    causal = causal[None, None]

    g_full = graph(emb, full, "full")
    g_causal = graph(emb, causal, "causal")

    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    C = os.path.join(P.MODEL)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                      num_hidden_layers=28, num_attention_heads=16,
                      num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
                      rope_theta=1000000, max_position_embeddings=40960,
                      tie_word_embeddings=False, attention_bias=False,
                      hidden_act="silu")
    model = Qwen3Model(cfg).to(torch.float32)
    sd = {}
    with safe_open(os.path.join(C, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("llm.") and "embed_tokens" not in k:
                sd[k[4:]] = f.get_tensor(k).float()
    model.load_state_dict(sd, strict=False)
    model.eval()
    x = torch.from_numpy(emb)
    with torch.no_grad():
        t_full = model(inputs_embeds=x, attention_mask=torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bool)).last_hidden_state.numpy().astype(np.float64)
        t_causal = model(inputs_embeds=x, attention_mask=torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))[None, None]).last_hidden_state.numpy().astype(np.float64)
    print("torch full vs causal: cos %.4f" % cos(t_full, t_causal), flush=True)
    for name, g in (("full", g_full), ("causal", g_causal)):
        for tn, t in (("torch-full", t_full), ("torch-causal", t_causal)):
            rows = slice(34, 256)
            print(f"graph-{name} vs {tn}: cos(all) %.4f  cos(rows34+) %.4f"
                  % (cos(g, t), cos(g[rows], t[rows])), flush=True)


if __name__ == "__main__":
    main()
