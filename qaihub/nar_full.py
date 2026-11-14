#!/usr/bin/env python3
"""NAR with the full 256-token window (frames = 256 - cond_len).

The compiled backbone ignores attention_mask (four different masks give
bit-identical output), so it always attends all 256 slots.  Padding with
mask-token text embeddings poisons the attention, so instead we ask the model
for exactly as many target frames as fit: cond_len + frames == 256 and there is
no pad region at all -- which is exactly what the reference does at batch=1.

usage: nar_full.py <text> <out.wav> [dec_frames]
       dec_frames = how many of the generated frames to decode (default 150,
       the most the fixed 200-frame decoder window can take with 25+25 silence).

Env: VV_LLM (context binary), VV_STEPS, VV_TEMP
"""
import math
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "你好，世界。今天我们测试一下这个语音合成系统。"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(P.HERE, "nar_full.wav")
    dec_n = int(sys.argv[3]) if len(sys.argv) > 3 else 150

    tab = P.Tables()
    # cond length does not depend on the frame count, so probe once.
    _, _, _, cond, _ = P.build_prompt(tab, text, "zh", 1)
    frames = P.SEQ - cond
    ids, am, attn, cond, total = P.build_prompt(tab, text, "zh", frames)
    est = P.estimate_frames(text)
    print(f"[prompt] cond {cond} + target {frames} = {total} / {P.SEQ} "
          f"(estimate {est} frames)", flush=True)

    attn.tofile(os.path.join(P.WORK, "attn.raw"))
    rng = np.random.default_rng(0)
    mask_ids = np.array(P.MASK_IDS, dtype=np.int64)
    tokens = mask_ids[:, None].repeat(frames, axis=1)

    u = np.linspace(0.0, 1.0, P.STEPS + 1)
    ts = P.T_SHIFT * u / (1.0 + (P.T_SHIFT - 1.0) * u)
    total_mask = frames * P.N_CB
    remaining = total_mask
    schedule = []
    for s in range(P.STEPS):
        k = remaining if s == P.STEPS - 1 else min(
            math.ceil(total_mask * (ts[s + 1] - ts[s])), remaining)
        schedule.append(int(k))
        remaining -= int(k)

    layer_ids = np.arange(P.N_CB)[:, None].astype(np.float32)
    t_all = time.perf_counter()
    llm_ms = head_ms = 0.0
    for step in range(P.STEPS):
        ids[:, cond:cond + frames] = tokens
        emb = P.build_embeds(tab, ids, am)
        emb.tofile(os.path.join(P.WORK, "emb.raw"))
        (hid,), ms = P.net_run(P.LLM_BIN,
                               [os.path.join(P.WORK, "emb.raw"),
                                os.path.join(P.WORK, "attn.raw")],
                               os.path.join(P.WORK, "llm"),
                               [("output_0.raw", P.SEQ * 1024 * 4)])
        llm_ms += ms
        hid.astype(np.float32).tofile(os.path.join(P.WORK, "hid.raw"))
        lg, ms = P.net_run(P.HEAD_BIN, [os.path.join(P.WORK, "hid.raw")],
                           os.path.join(P.WORK, "head"),
                           [("output_0.raw", P.SEQ * P.VOCAB_SEM * 4),
                            ("output_1.raw", P.SEQ * 11 * P.VOCAB_ACO * 4)])
        head_ms += ms
        sem = lg[0].reshape(P.SEQ, P.VOCAB_SEM)[cond:cond + frames].astype(np.float64)
        aco = lg[1].reshape(P.SEQ, 11, P.VOCAB_ACO)[cond:cond + frames].astype(np.float64)

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
        pred = np.empty((P.N_CB, frames), dtype=np.int64)
        scores = np.empty((P.N_CB, frames), dtype=np.float32)
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
            tokens = fw.reshape(P.N_CB, frames)
            hits = top[top < frames]
            pz = hits[pred[0, hits] == P.PAUSE_IDS[0]]
            if pz.size:
                tokens[:, pz] = np.array(P.PAUSE_IDS, dtype=np.int64)[:, None]
        if step % 8 == 0 or step == P.STEPS - 1:
            print(f"[step {step:2d}] fill {k:5d} filled "
                  f"{(tokens != mask_ids[:, None]).sum()}/{total_mask}", flush=True)

    print(f"[nar] {time.perf_counter()-t_all:.1f}s llm {llm_ms/1000:.1f}s "
          f"head {head_ms/1000:.1f}s", flush=True)
    np.save(os.path.join(P.WORK, "full_tokens.npy"), tokens)
    sem = tokens[0]
    print(f"[nar] sem uniques {len(set(sem.tolist()))} first 30 "
          f"{sem[:30].tolist()}", flush=True)

    dec_n = min(dec_n, frames, P.FRAMES - 50)
    P.decoder(tokens[:, :dec_n], out)

    w = wave.open(out)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    n = 2400
    env = [float(np.sqrt((a[i:i+n]**2).mean())) for i in range(0, len(a) - n, n)]
    print("[out] %.2fs rms %.4f" % (len(a)/24000, float(np.sqrt((a**2).mean()))), flush=True)
    print("[out] env", " ".join("%.2f" % x for x in env[:60]), flush=True)


if __name__ == "__main__":
    main()
