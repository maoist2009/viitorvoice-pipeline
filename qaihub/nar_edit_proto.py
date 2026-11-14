#!/usr/bin/env python3
"""Local-edit prototype on the phone NPU: masked regeneration of a frame span.

Mirrors viitorvoice/llm/generation.py::_generate_edit_iterative_single for the
single-utterance / no-ref-audio case:

    branch "full"    prompt(style+text) + window tokens   (editable frames masked)
    branch "uncond"  window tokens only
    guided = log_softmax(logp_c) + cfg * (logp_c - logp_u)   -> log_softmax

Everything else (schedule, layer penalty, gumbel, pause collapse) is the same
as the synthesis loop in nar_proto.py, except that the score mask only lets the
*editable* frames update and the schedule is built from the editable cell count.

usage: nar_edit_proto.py [--text A] [--edit B] [--frames N] [--mask a,b]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as NP  # noqa: E402

CFG = 2.0
STEPS = NP.STEPS
T_SHIFT = NP.T_SHIFT
LAYER_PENALTY = NP.LAYER_PENALTY
POS_TEMP = NP.TEMP


def prompt_ids(t, text, lang="zh"):
    """style + <|text_start|>...<|text_end|> ids, exactly like build_prompt()."""
    style = (f"<|lang_start|>{lang}<|lang_end|>"
             f"<|instruct_start|>None<|instruct_end|>")
    style_ids = NP.tokenize_with_special_tokens(style, t.tok).tolist()[0]
    full = NP.combine_text(text=text, ref_text=None)
    full = NP.prepare_text_for_tokenizer(full, t.tok, lang)
    full = NP.insert_text_pause_anchors(full)
    text_ids = NP.tokenize_with_special_tokens(f"<|text_start|>{full}<|text_end|>",
                                               t.tok).tolist()[0]
    return style_ids + text_ids


def make_branch(t, cond, window, editable, uncond):
    """ids [12,256], audio_mask [256], attn [1,1,256,256] for one branch."""
    W = window.shape[1]
    ids = np.zeros((NP.N_CB, NP.SEQ), dtype=np.int64)
    ids[:, :] = np.array(NP.MASK_IDS, dtype=np.int64)[:, None]
    audio_mask = np.zeros(NP.SEQ, dtype=bool)
    attn = np.zeros((1, 1, NP.SEQ, NP.SEQ), dtype=np.float32)
    if uncond:
        ids[:, :W] = window
        audio_mask[:W] = True
        total = W
    else:
        ids[:, :len(cond)] = np.repeat(np.array(cond, dtype=np.int64)[None, :], NP.N_CB, axis=0)
        ids[:, len(cond):len(cond) + W] = window
        audio_mask[len(cond):len(cond) + W] = True
        total = len(cond) + W
    attn[0, 0, :total, :total] = 1.0
    idx = np.arange(total, NP.SEQ)
    attn[0, 0, idx, idx] = 1.0
    return ids, audio_mask, attn, len(cond) if not uncond else 0


def run_llm_head(t, ids, audio_mask, attn, tag, d):
    emb = NP.build_embeds(t, ids, audio_mask)
    os.makedirs(d, exist_ok=True)
    np.ascontiguousarray(emb).tofile(f"{d}/emb_{tag}.raw")
    np.ascontiguousarray(attn).tofile(f"{d}/attn_{tag}.raw")
    outs, dt1 = NP.net_run(NP.LLM_BIN, [f"{d}/emb_{tag}.raw", f"{d}/attn_{tag}.raw"],
                           f"{d}/llm_{tag}", [("hidden", NP.SEQ * 1024 * 4)])
    hidden = outs[0]
    hidden.tofile(f"{d}/hidden_{tag}.raw")
    outs, dt2 = NP.net_run(NP.HEAD_BIN, [f"{d}/hidden_{tag}.raw"], f"{d}/head_{tag}",
                           [("semantic_logits", NP.SEQ * NP.VOCAB_SEM * 4),
                            ("acoustic_logits", NP.SEQ * 11 * NP.VOCAB_ACO * 4)])
    sem = outs[0].reshape(NP.SEQ, NP.VOCAB_SEM)
    aco = outs[1].reshape(NP.SEQ, 11, NP.VOCAB_ACO)
    return sem, aco, dt1 + dt2


def log_softmax(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return x - m - np.log(e.sum(-1, keepdims=True))


def schedule(n_cells, steps=STEPS):
    ts = [T_SHIFT * (i / steps) / (1.0 + (T_SHIFT - 1.0) * (i / steps)) for i in range(steps + 1)]
    out, rem = [], n_cells
    for s in range(steps):
        num = rem if s == steps - 1 else min(math.ceil(n_cells * (ts[s + 1] - ts[s])), rem)
        out.append(int(num))
        rem -= int(num)
    return out


def rng_gumbel(rng):
    return -np.log(-np.log(rng.random() + 1e-10) + 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="今天我们来测试一下语音编辑功能。")
    ap.add_argument("--edit", default="今天我们来体验一下语音编辑功能。")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--mask", default="")
    ap.add_argument("--cfg", type=float, default=CFG)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tag", default="edit")
    a = ap.parse_args()

    t = NP.Tables()
    frames = a.frames or NP.estimate_frames(a.text)
    print(f"[plan] {len(a.text)} chars -> {frames} frames ({frames/25:.2f}s)", flush=True)

    # ------------------------------------------------ 1. synthesise the draft
    cond = prompt_ids(t, a.text)
    ids, am, attn, clen, total = NP.build_prompt(t, a.text, "zh", frames)
    tokens = np.tile(np.array(NP.MASK_IDS, dtype=np.int64)[:, None], (1, frames))
    sem_mask = np.zeros(NP.VOCAB_SEM, bool); sem_mask[NP.MASK_IDS[0]] = sem_mask[NP.SEP_IDS[0]] = True
    aco_mask = np.zeros(NP.VOCAB_ACO, bool); aco_mask[NP.MASK_IDS[1]] = aco_mask[NP.SEP_IDS[1]] = True
    mask_ids = np.array(NP.MASK_IDS, dtype=np.int64)[:, None]
    layer_ids = np.arange(NP.N_CB)[:, None]
    sched = schedule(NP.N_CB * frames, a.steps)
    rng = np.random.default_rng(1234)
    d = f"{NP.WORK}/{a.tag}_work"
    t0 = time.time()
    for step in range(a.steps):
        ids[:, clen:clen + frames] = tokens
        sem, aco, _ = run_llm_head(t, ids, am, attn, "c", d)
        rows = slice(clen, clen + frames)
        s = sem[rows]; ac = aco[rows]
        ls = log_softmax(s); la = log_softmax(ac)
        ls[..., sem_mask] = -np.inf
        la[..., aco_mask] = -np.inf
        pred = np.concatenate([ls.argmax(-1)[:, None], la.argmax(-1)], axis=1)   # (T, 12)
        sc = np.concatenate([ls.max(-1)[:, None], la.max(-1)], axis=1)          # (T, 12)
        sc = sc - layer_ids.T * LAYER_PENALTY
        sc = sc / POS_TEMP + rng_gumbel(rng)
        alive = (tokens == mask_ids).T                                          # (T, 12)
        sc = np.where(alive, sc, -np.inf)
        k = min(sched[step], int(np.isfinite(sc).sum()))
        if k <= 0:
            continue
        flat = sc.ravel()
        idx = np.argpartition(-flat, k - 1)[:k]
        py, pc = np.unravel_index(idx, sc.shape)
        tokens[pc, py] = pred[py, pc]
        # pause collapse on the semantic row
        for j in range(k):
            if pc[j] == 0 and pred[py[j], 0] == NP.PAUSE_IDS[0]:
                tokens[:, py[j]] = np.array(NP.PAUSE_IDS)
        print(f"  step {step:2d} k={k:3d} left={int(np.isfinite(sc).sum()):4d} "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{NP.WORK}/{a.tag}_draft.npy", tokens)
    NP.decoder(tokens, f"{NP.WORK}/{a.tag}_draft.wav")
    print("[ok] draft ->", f"{NP.WORK}/{a.tag}_draft.wav", flush=True)

    # ------------------------------------------------ 2. edit a span with CFG
    if a.mask:
        m0, m1 = [int(x) for x in a.mask.split(",")]
    else:
        # the changed characters, mapped proportionally onto the frames
        i0 = next(i for i in range(min(len(a.text), len(a.edit))) if a.text[i] != a.edit[i])
        j = 0
        while j < min(len(a.text), len(a.edit)) and a.text[-1 - j] == a.edit[-1 - j]:
            j += 1
        i1 = len(a.edit) - j
        m0 = max(0, int(i0 / len(a.text) * frames) - 2)
        m1 = min(frames, int(math.ceil(i1 / len(a.text) * frames)) + 2)
    print(f"[edit] mask frames [{m0},{m1}) of {frames}", flush=True)

    cond_e = prompt_ids(t, a.edit)
    editable = np.zeros(frames, bool); editable[m0:m1] = True
    win = tokens.copy()
    win[:, editable] = mask_ids
    ids_c, am_c, attn_c, clen_c = make_branch(t, cond_e, win, editable, False)
    ids_u, am_u, attn_u, _ = make_branch(t, cond_e, win, editable, True)
    n_cells = int(editable.sum()) * NP.N_CB
    sched = schedule(n_cells, a.steps)
    t0 = time.time()
    remaining = n_cells
    for step in range(a.steps):
        ids_c[:, clen_c:clen_c + frames] = win
        ids_u[:, :frames] = win
        sem_c, aco_c, _ = run_llm_head(t, ids_c, am_c, attn_c, "c", d)
        sem_u, aco_u, _ = run_llm_head(t, ids_u, am_u, attn_u, "u", d)
        c_s = sem_c[clen_c:clen_c + frames]; u_s = sem_u[:frames]
        c_a = aco_c[clen_c:clen_c + frames]; u_a = aco_u[:frames]
        lc_s, lu_s = log_softmax(c_s), log_softmax(u_s)
        lc_a, lu_a = log_softmax(c_a), log_softmax(u_a)
        g_s = log_softmax(lc_s + a.cfg * (lc_s - lu_s))
        g_a = log_softmax(lc_a + a.cfg * (lc_a - lu_a))
        g_s[..., sem_mask] = -np.inf
        g_a[..., aco_mask] = -np.inf
        pred = np.concatenate([g_s.argmax(-1)[:, None], g_a.argmax(-1)], axis=1)
        sc = np.concatenate([g_s.max(-1)[:, None], g_a.max(-1)], axis=1)
        sc = sc - layer_ids.T * LAYER_PENALTY
        sc = sc / POS_TEMP + rng_gumbel(rng)
        alive = (win == mask_ids).T & editable[:, None]
        sc = np.where(alive, sc, -np.inf)
        k = min(sched[step], int(np.isfinite(sc).sum()))
        if k <= 0:
            print(f"  edit step {step:2d} nothing left", flush=True)
            continue
        flat = sc.ravel()
        idx = np.argpartition(-flat, k - 1)[:k]
        py, pc = np.unravel_index(idx, sc.shape)
        win[pc, py] = pred[py, pc]
        for j in range(k):
            if pc[j] == 0 and pred[py[j], 0] == NP.PAUSE_IDS[0]:
                win[:, py[j]] = np.array(NP.PAUSE_IDS)
        win[:, ~editable] = tokens[:, ~editable]
        remaining -= k
        print(f"  edit step {step:2d} k={k:3d} left={max(0,remaining):4d} "
              f"({time.time()-t0:.0f}s)", flush=True)

    left = int((win[:, editable] == mask_ids).sum())
    print(f"[edit] done, unfilled mask cells: {left}", flush=True)
    np.save(f"{NP.WORK}/{a.tag}_tokens.npy", win)
    NP.decoder(win, f"{NP.WORK}/{a.tag}_after.wav")
    print("[ok] edited ->", f"{NP.WORK}/{a.tag}_after.wav", flush=True)
    print(json.dumps({"mask": [m0, m1], "frames": frames, "cfg": a.cfg}))


if __name__ == "__main__":
    main()
