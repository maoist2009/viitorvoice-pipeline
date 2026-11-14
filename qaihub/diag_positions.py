#!/usr/bin/env python3
"""Where does the fp16 DLC start to diverge?

With a *causal* mask, token 0's hidden state is a function of emb[0] alone, so
"cos(pos 0)" isolates the per-token path (RMSNorm+MLP+self-attention) from the
cross-token path.  A second run forces every position to attend to position 0
only, which exercises the mask plumbing hard.

usage: diag_positions.py
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
FP16 = os.path.join(P.HERE, "fp16/job_jgnq1k3vg_linked_bin_mqy33gjor.bin")
NEG = -30000.0


def graph(emb, add_mask, tag):
    emb.astype(np.float32).tofile(os.path.join(W, "emb.raw"))
    add_mask.astype(np.float32).tofile(os.path.join(W, "attn_dp.raw"))
    (hid,), ms = P.net_run(FP16, [os.path.join(W, "emb.raw"),
                                  os.path.join(W, "attn_dp.raw")],
                           os.path.join(W, "llm_dp_" + tag),
                           [("output_0.raw", SEQ * DIM * 4)])
    print(f"[graph {tag}] {ms:.0f} ms", flush=True)
    return hid.astype(np.float64)


def torch_ref(emb, bool_mask):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    C = P.MODEL
    cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                      num_hidden_layers=28, num_attention_heads=16,
                      num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
                      rope_theta=1000000, max_position_embeddings=40960,
                      tie_word_embeddings=False, attention_bias=False,
                      hidden_act="silu")
    cfg._attn_implementation = "eager"
    model = Qwen3Model(cfg).to(torch.float32)
    sd = {}
    with safe_open(os.path.join(C, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("llm.") and "embed_tokens" not in k:
                sd[k[4:]] = f.get_tensor(k).float()
    model.load_state_dict(sd, strict=False)
    model.eval()
    with torch.no_grad():
        out = model(inputs_embeds=torch.from_numpy(emb.astype(np.float32)),
                    attention_mask=bool_mask).last_hidden_state
    return out.numpy()[0].astype(np.float64)


def cos_rows(a, b):
    num = (a * b).sum(-1)
    return num / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def report(name, g, t):
    c = cos_rows(g, t)
    print(f"\n== {name}: cos mean {c.mean():.4f}")
    for i in (0, 1, 2, 3, 7, 15, 31, 63, 127, 191, 255):
        print(f"   pos {i:3d}: cos {c[i]:.4f}")
    print("   blocks of 32:", " ".join(f"{c[i:i+32].mean():.3f}" for i in range(0, 256, 32)))


def main():
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    emb = np.load(os.path.join(W, "emb256.npy")).astype(np.float32)
    i = np.arange(SEQ)
    bool_tril = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))[None, None]
    add_tril = np.where(i[None, :] > i[:, None], NEG, 0.0).astype(np.float32)[None, None]

    t = torch_ref(emb, bool_tril)
    g = graph(emb, add_tril, "tril").reshape(SEQ, DIM)
    report("causal (tril)", g, t)
    np.save(os.path.join(W, "torch_tril.npy"), t)
    np.save(os.path.join(W, "graph_tril.npy"), g)

    # every position may only attend to position 0
    bool_pin = torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bool)
    bool_pin[:, :, :, 0] = True
    add_pin = np.full((1, 1, SEQ, SEQ), NEG, dtype=np.float32)
    add_pin[:, :, :, 0] = 0.0
    t2 = torch_ref(emb, bool_pin)
    g2 = graph(emb, add_pin, "pin0").reshape(SEQ, DIM)
    report("attend-to-0-only", g2, t2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
