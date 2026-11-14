#!/usr/bin/env python3
"""NAR with the fixed 256-token window filled with *silence codec frames*.

The compiled backbone ignores attention_mask (verified: four different masks
give bit-identical output), so it always attends every one of the 256 slots.
Whenever cond_len + frames < 256 the leftover slots sit there as mask-token
text embeddings and poison the attention.  Filling them with real dualcodec
silence frames (from assets/dualcodec_silence_2s.pt) keeps the prefix-suffix
silence structure the decoder was trained with and stops the model from
attending to junk.

usage: nar_fill.py <text> <out.wav> [frames]   (frames=0 -> duration estimate)
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402

SIL = ("/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/"
       "viitor-voice-nar/local_models/assets/dualcodec_silence_2s.pt")


def load_silence():
    d = torch.load(SIL, map_location="cpu")
    t = torch.as_tensor(d["tokens"] if isinstance(d, dict) else d).long().numpy()
    return t.astype(np.int64)


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "你好，世界。今天我们测试一下这个语音合成系统。"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(P.HERE, "nar_fill.wav")
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    tab = P.Tables()
    cond_ids = None
    if frames <= 0:
        frames = P.estimate_frames(text)
    ids, am, attn, cond, total = P.build_prompt(tab, text, "zh", frames)
    print(f"[prompt] cond {cond} + target {frames} = {total} / {P.SEQ}", flush=True)

    sil = load_silence()
    sil_pat = sil
    if total < P.SEQ:
        need = P.SEQ - total
        reps = int(np.ceil(need / sil_pat.shape[1]))
        filler = np.tile(sil_pat, (1, reps))[:, :need]
        ids[:, total:] = filler
        am[total:] = True
    # the filler frames are context only: freeze them
    fixed = np.zeros(P.SEQ, dtype=bool)
    fixed[total:] = True
    fixed[:cond] = True

    attn[:] = 1.0
    rng = np.random.default_rng(0)
    mask_ids = np.array(P.MASK_IDS, dtype=np.int64)
    tokens = mask_ids[:, None].repeat(frames, axis=1)
    edit = np.zeros((P.N_CB, P.SEQ), dtype=bool)
    edit[:, cond:cond + frames] = True

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
    attn.tofile(os.path.join(P.WORK, "attn.raw"))
    t_all = time.perf_counter()
    llm_ms = head_ms = 0.0
    for step in range(P.STEPS):
        ids[:, cond:cond + frames] = tokens
        emb = P.build_embeds(tab, ids, am)
        emb.tofile(os.path.join(P.WORK, "emb.raw"))
        (hid,), ms = P.net_run(P.LLM_BIN, [os.path.join(P.WORK, "emb.raw"),
                                           os.path.join(P.WORK, "attn.raw")],
                               os.path.join(P.WORK, "llm"), [("output_0.raw", P.SEQ * 1024 * 4)])
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
        print(f"[step {step:2d}] fill {k:5d} filled {(tokens != mask_ids[:, None]).sum()}/{total_mask}",
              flush=True)

    print(f"[nar] {time.perf_counter()-t_all:.1f}s llm {llm_ms/1000:.1f}s head {head_ms/1000:.1f}s",
          flush=True)
    np.save(os.path.join(P.WORK, "fill_tokens.npy"), tokens)

    # decode: 25 silence frames on each side, silence-padded up to the fixed
    # 200-frame decoder window, then trim back to the real target.
    gap = sil[:, :25]
    body = tokens
    packed = np.concatenate([gap, body, gap], axis=1)
    if packed.shape[1] < P.FRAMES:
        need = P.FRAMES - packed.shape[1]
        reps = int(np.ceil(need / sil.shape[1]))
        packed = np.concatenate([packed, np.tile(sil, (1, reps))[:, :need]], axis=1)
    packed = packed[:, :P.FRAMES]
    print(f"[dec] packed {packed.shape[1]} frames, target {frames}", flush=True)
    P.FRAMES = P.FRAMES
    wav = P.decoder_raw(packed, out)
    # trim lead gap and tail
    import wave
    w = wave.open(out)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    lead = 25 * 960
    seg = a[lead:lead + frames * 960]
    pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16)
    with wave.open(out, "wb") as ww:
        ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(24000)
        ww.writeframes(pcm.tobytes())
    n = 2400
    env = [float(np.sqrt((seg[i:i+n]**2).mean())) for i in range(0, len(seg) - n, n)]
    print("[out] %.2fs rms %.4f" % (len(seg) / 24000, float(np.sqrt((seg**2).mean()))), flush=True)
    print("[out] env", " ".join("%.2f" % x for x in env[:40]), flush=True)


if __name__ == "__main__":
    main()
