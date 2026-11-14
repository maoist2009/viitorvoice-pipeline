#!/usr/bin/env python3
"""Full NAR loop with the torch backbone — the ground truth for the C++ port.

Same math as nar_proto.py (32 steps, t_shift 0.1, layer penalty 5, position
temperature 5, no CFG) but the backbone runs in torch, so it can attend
bidirectionally the way generation.py does. The compiled graph cannot: its
attention_mask input is dead (verified bit-identical output for four different
masks), so it must be running whatever mask the ONNX export baked in.

usage: torch_pipeline.py <text> <out.wav> [frames] [--causal]
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402

CTX = "/data/data/com.termux/files/usr/tmp/vv_nar"


def build_model():
    cfg_json = json.load(open(os.path.join(P.MODEL, "config.json")))["llm_config"]
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    cfg = Qwen3Config(
        vocab_size=cfg_json["vocab_size"], hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        rms_norm_eps=1e-6, rope_theta=1000000, max_position_embeddings=40960,
        tie_word_embeddings=False, attention_bias=False, hidden_act="silu")
    model = Qwen3Model(cfg).to(torch.bfloat16)
    sd = {}
    with safe_open(os.path.join(P.MODEL, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("llm.") and not k[4:].startswith("embed_tokens"):
                sd[k[4:]] = f.get_tensor(k).to(torch.bfloat16)
    model.load_state_dict(sd, strict=False)
    model.eval()
    with safe_open(os.path.join(P.MODEL, "model.safetensors"), framework="np") as f:
        sem_w = torch.from_numpy(f.get_tensor("semantic_head.weight").astype(np.float32))
        aco_w = torch.from_numpy(f.get_tensor("acoustic_head.weight").astype(np.float32))
    return model, sem_w, aco_w


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    causal = "--causal" in sys.argv
    text = args[0] if args else "你好，世界。今天我们测试一下这个语音合成系统。"
    out = args[1] if len(args) > 1 else os.path.join(P.HERE, "torch_out.wav")
    frames = int(args[2]) if len(args) > 2 else 200

    torch.set_num_threads(6)
    torch.set_grad_enabled(False)

    t = P.Tables()
    ids, audio_mask, attn, cond_len, total = P.build_prompt(t, text, "zh", frames)
    print(f"[prompt] cond {cond_len} + {frames} = {total}", flush=True)

    t0 = time.perf_counter()
    model, sem_w, aco_w = build_model()
    print(f"[torch] model loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    if causal:
        tril = np.tril(np.ones((P.SEQ, P.SEQ), dtype=np.float32))
        attn = tril[None, None]
        print("[torch] using causal mask", flush=True)
    mask = torch.from_numpy(attn > 0.5)

    rng = np.random.default_rng(0)
    mask_ids = np.array(P.MASK_IDS, dtype=np.int64)
    tokens = mask_ids[:, None].repeat(frames, axis=1)
    layer_ids = np.arange(12)[:, None].astype(np.float32)

    u = np.linspace(0.0, 1.0, P.STEPS + 1)
    ts = P.T_SHIFT * u / (1.0 + (P.T_SHIFT - 1.0) * u)
    total_mask = frames * 12
    remaining = total_mask
    schedule = []
    for s in range(P.STEPS):
        k = remaining if s == P.STEPS - 1 else min(
            math.ceil(total_mask * (ts[s + 1] - ts[s])), remaining)
        schedule.append(int(k))
        remaining -= int(k)
    print("[torch] schedule", schedule, flush=True)

    t_all = time.perf_counter()
    for step in range(P.STEPS):
        ids[:, cond_len:total] = tokens
        emb = P.build_embeds(t, ids, audio_mask)
        x = torch.from_numpy(emb).to(torch.bfloat16)
        h = model(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        hf = h[0].float()
        sem = (hf @ sem_w.T).numpy()[cond_len:total].astype(np.float64)
        aco = (hf @ aco_w.T).numpy()[cond_len:total].reshape(frames, 11, 1027).astype(np.float64)

        def logp(x):
            m = x.max(-1, keepdims=True)
            e = np.exp(x - m)
            return (x - m) - np.log(e.sum(-1, keepdims=True))

        sl = logp(sem)
        al = logp(aco)
        sl[:, P.MASK_IDS[0]] = -np.inf
        sl[:, P.SEP_IDS[0]] = -np.inf
        al[:, :, P.MASK_IDS[1]] = -np.inf
        al[:, :, P.SEP_IDS[1]] = -np.inf
        pred = np.empty((12, frames), dtype=np.int64)
        scores = np.empty((12, frames), dtype=np.float32)
        pred[0] = sl.argmax(-1)
        scores[0] = sl.max(-1)
        for c in range(11):
            pred[c + 1] = al[:, c].argmax(-1)
            scores[c + 1] = al[:, c].max(-1)
        scores = scores - layer_ids * P.LAYER_PENALTY
        g = -np.log(-np.log(rng.random(scores.shape) + 1e-10) + 1e-10)
        scores = scores / P.TEMP + g
        scores[tokens != mask_ids[:, None]] = -np.inf
        k = schedule[step]
        if k > 0:
            flat = scores.reshape(-1)
            top = np.argpartition(-flat, k - 1)[:k]
            fw = tokens.reshape(-1)
            fw[top] = pred.reshape(-1)[top]
            tokens = fw.reshape(12, frames)
            hits = top[top < frames]
            pz = hits[pred[0, hits] == P.PAUSE_IDS[0]]
            if pz.size:
                tokens[:, pz] = np.array(P.PAUSE_IDS, dtype=np.int64)[:, None]
        print(f"[step {step:2d}] fill {k:5d} filled {(tokens != mask_ids[:, None]).sum()}/{total_mask}",
              flush=True)

    print(f"[nar] {time.perf_counter()-t_all:.0f}s", flush=True)
    np.save(os.path.join(CTX, "torch_tokens.npy"), tokens)
    P.decoder(tokens, out)
    print("[done]", out, os.path.getsize(out), "B", flush=True)


if __name__ == "__main__":
    main()
