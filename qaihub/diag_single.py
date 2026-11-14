#!/usr/bin/env python3
"""Where does the compiled backbone diverge?

Three probes, graph vs torch fp32 on the same embeds:

 1. attention pinned to key 0 (every query may only look at position 0).  The
    attention output becomes the same V_0 for every row, so this measures the
    MLP / norm / residual path with content mixing switched off.
 2. causal mask, to see whether the graph's mask handling matches torch's.
 3. permuted input rows, to calibrate how much a content permutation moves the
    output (the graph only moved to cos 0.63).
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


def c(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def graph(emb, attn, tag):
    emb.astype(np.float32).tofile(os.path.join(W, "emb.raw"))
    attn.astype(np.float32).tofile(os.path.join(W, "attn_%s.raw" % tag))
    (h,), ms = P.net_run(FP16, [os.path.join(W, "emb.raw"),
                                os.path.join(W, "attn_%s.raw" % tag)],
                         os.path.join(W, "llm_" + tag), [("output_0.raw", SEQ * DIM * 4)])
    return h.reshape(SEQ, DIM).astype(np.float64)


def main():
    emb = np.load(os.path.join(W, "emb256.npy")).astype(np.float32).reshape(SEQ, DIM)
    i = np.arange(SEQ)

    add_pin0 = np.full((1, 1, SEQ, SEQ), -30000.0, np.float32)
    add_pin0[0, 0, :, 0] = 0.0
    add_causal = np.where(i[None, :] > i[:, None], -30000.0, 0.0).astype(np.float32)[None, None]
    add_full = np.zeros((1, 1, SEQ, SEQ), np.float32)

    perm = np.random.default_rng(7).permutation(SEQ)
    g = {}
    g["pin0"] = graph(emb, add_pin0, "pin0")
    g["causal"] = graph(emb, add_causal, "causal")
    g["full"] = graph(emb, add_full, "full")
    g["perm"] = graph(emb[perm], add_full, "perm")

    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                      num_hidden_layers=28, num_attention_heads=16,
                      num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
                      rope_theta=1000000, max_position_embeddings=40960,
                      tie_word_embeddings=False, attention_bias=False, hidden_act="silu")
    model = Qwen3Model(cfg).to(torch.float32)
    sd = {}
    with safe_open(os.path.join(P.MODEL, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("llm.") and "embed_tokens" not in k:
                sd[k[4:]] = f.get_tensor(k).float()
    model.load_state_dict(sd, strict=False)
    model.eval()
    x = torch.from_numpy(emb)

    def t(m):
        with torch.no_grad():
            return model(inputs_embeds=x if m is not None else x, attention_mask=m).last_hidden_state.numpy().reshape(SEQ, DIM).astype(np.float64)

    pin = np.zeros((1, 1, SEQ, SEQ), bool)
    pin[0, 0, :, 0] = True
    t_pin = t(torch.from_numpy(pin))
    t_causal = t(torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))[None, None])
    t_full = t(torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bool))
    with torch.no_grad():
        t_perm = model(inputs_embeds=x[torch.from_numpy(perm)],
                       attention_mask=torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bool)).last_hidden_state.numpy().reshape(SEQ, DIM).astype(np.float64)

    print("torch pin0  std %.4f | graph pin0  std %.4f | cos %.4f" %
          (t_pin.std(), g["pin0"].std(), c(t_pin, g["pin0"])), flush=True)
    print("torch causal cos %.4f | graph causal cos %.4f" %
          (c(t_full, t_causal), c(g["full"], g["causal"])), flush=True)
    print("cos graph-full vs torch-full  %.4f" % c(g["full"], t_full), flush=True)
    print("cos graph-causal vs torch-causal %.4f" % c(g["causal"], t_causal), flush=True)
    print("perm: graph %.4f | torch %.4f (vs own unpermuted)" %
          (c(g["perm"][np.argsort(perm)], g["full"]), c(t_perm[np.argsort(perm)], t_full)), flush=True)


if __name__ == "__main__":
    main()
