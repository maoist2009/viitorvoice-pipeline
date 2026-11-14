#!/usr/bin/env python3
"""CPU reference for the ViiTorVoice backbone + heads.

The compiled graphs are hard to inspect, so this rebuilds the same forward pass
from the checkpoint with transformers' Qwen3 (bf16 to keep the phone alive) and
compares, step by step, against what the NPU produced:

  * embeddings: host fp16 tables vs the fp32 checkpoint rows
  * backbone:   NPU hidden vs torch hidden
  * heads:      NPU logits vs a torch matmul on the same hidden

usage: torch_ref.py <emb.raw> <attn.raw> <npu_hidden.raw> [out_prefix]
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

CKPT = os.path.expanduser(
    "~/projects/voice_editor/sm8750_voice_editor/viitor-voice-nar/"
    "local_models/llm/0p6_emotion")
SEQ, DIM = 256, 1024


def main():
    emb_path, attn_path, hid_path = sys.argv[1:4]
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)

    cfg_json = json.load(open(os.path.join(CKPT, "config.json")))["llm_config"]
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    cfg = Qwen3Config(
        vocab_size=cfg_json["vocab_size"],
        hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        rms_norm_eps=1e-6, rope_theta=1000000, max_position_embeddings=40960,
        tie_word_embeddings=False, attention_bias=False, hidden_act="silu",
    )
    dt = torch.float32 if os.environ.get("VV_DTYPE", "fp32") == "fp32" else torch.bfloat16
    model = Qwen3Model(cfg).to(dt)
    print("model built, params %.0f M" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    sd = {}
    with safe_open(os.path.join(CKPT, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if not k.startswith("llm."):
                continue
            name = k[4:]
            if name.startswith("embed_tokens"):
                continue            # we feed inputs_embeds directly
            sd[name] = f.get_tensor(k).to(dt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing", len(missing), "unexpected", len(unexpected), flush=True)
    if missing:
        print(" missing sample:", missing[:4])

    emb = np.fromfile(emb_path, dtype=np.float32).reshape(1, SEQ, DIM)
    attn = np.fromfile(attn_path, dtype=np.float32).reshape(1, 1, SEQ, SEQ)
    x = torch.from_numpy(emb).to(dt)
    mask = torch.from_numpy(attn > 0.5)
    out = model(inputs_embeds=x, attention_mask=mask, use_cache=False)
    h = out.last_hidden_state.float().numpy().reshape(SEQ, DIM)
    print("torch hidden std %.4f absmax %.2f" % (h.std(), np.abs(h).max()), flush=True)

    npu = np.fromfile(hid_path, dtype=np.float32).reshape(SEQ, DIM)
    np.save("/data/data/com.termux/files/usr/tmp/vv_nar/torch_hid.npy", h)
    a = h[33:233].reshape(-1)
    b = npu[33:233].reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print("hidden target rows: cos %.4f  relL2 %.3f" % (cos, rel), flush=True)
    print("hidden std torch %.4f npu %.4f" % (h[33:233].std(), npu[33:233].std()), flush=True)
    tc = np.sum(h[33:233]*npu[33:233], 1)/(np.linalg.norm(h[33:233],axis=1)*np.linalg.norm(npu[33:233],axis=1))
    nr = np.linalg.norm(npu[33:233],axis=1)/np.linalg.norm(h[33:233],axis=1)
    print("per-token cos: mean %.4f min %.4f max %.4f" % (tc.mean(), tc.min(), tc.max()), flush=True)
    print("norm ratio npu/torch: mean %.3f std %.3f" % (nr.mean(), nr.std()), flush=True)

    with safe_open(os.path.join(CKPT, "model.safetensors"), framework="np") as f:
        sem_w = torch.from_numpy(f.get_tensor("semantic_head.weight").astype(np.float32))
        aco_w = torch.from_numpy(f.get_tensor("acoustic_head.weight").astype(np.float32))
    ht = torch.from_numpy(h)
    sem_logits = (ht @ sem_w.T).numpy()
    aco_logits = (ht @ aco_w.T).numpy()
    np.save("/data/data/com.termux/files/usr/tmp/vv_nar/ref_sem_logits.npy", sem_logits)
    np.save("/data/data/com.termux/files/usr/tmp/vv_nar/ref_aco_logits.npy", aco_logits)
    rows = slice(33, 233)
    pr = sem_logits[rows].argmax(1)
    print("torch sem argmax uniques %d, top count %d" % (len(np.unique(pr)), np.bincount(pr).max()), flush=True)
    print("torch sem first 20:", pr[:20], flush=True)


if __name__ == "__main__":
    main()
