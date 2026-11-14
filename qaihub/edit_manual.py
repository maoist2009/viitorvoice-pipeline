#!/usr/bin/env python3
"""Local edit with an *explicit* frame range (the "select a region of the
waveform, type new text" UX) using the official LLM edit path.

Mirrors what viitorvoice/local_edit does, but the region comes from a waveform
selection instead of a forced aligner:

    prefix frames = source[:, :start]
    masked block  = source[:, start:end]  -> replaced by N mask frames, where
                    N = RuleDurationEstimator(replacement_text) * 25
    suffix frames = source[:, end:]

then `generate_edit_masked` with the full edited sentence as the text condition.

usage:
    edit_manual.py <src_tokens.npy> <start_s> <end_s> <orig_span_text> \
                   <replacement_text> <full_text> <out_tokens.npy> [context_frames]
env: VV_DTYPE=fp32|bf16   VV_STEPS=32   VV_CFG=2.0   VV_DRYRUN=1
"""
import json
import os
import sys
import time

import numpy as np

IMPL = "/data/data/com.termux/files/home/projects/voice_editor/viitor-voice-nar"
LLM_DIR = ("/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/"
           "viitor-voice-nar/local_models/llm/0p6_emotion")
DL_VV = "/data/data/com.termux/files/home/projects/dreamlite/salvage/dl_vv"
FRAME_RATE = 25

sys.path.insert(0, IMPL)
sys.path.insert(0, DL_VV)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    a = sys.argv[1:]
    src_path, start_s, end_s, orig_span, repl, full_text, out_path = a[:7]
    ctx = int(a[7]) if len(a) > 7 else None

    import torch
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from viitorvoice.llm.generation import LLMGenerationConfig
    from viitorvoice.llm.text_utils import RuleDurationEstimator
    from torch_generate import build_generator

    src = np.load(src_path).astype(np.int64)
    total = src.shape[1]
    f0 = max(0, int(round(float(start_s) * FRAME_RATE)))
    f1 = min(total, int(round(float(end_s) * FRAME_RATE)))
    if f1 <= f0:
        raise SystemExit(f"empty selection {f0}:{f1}")

    est = RuleDurationEstimator()
    cfg = json.load(open(os.path.join(LLM_DIR, "config.json")))
    mask_ids = list(cfg["audio_mask_ids"])
    # keep the source's own speaking rate: same text weight => same frames
    w_old = est.calculate_total_weight(orig_span) if orig_span.strip() else 0.0
    w_new = est.calculate_total_weight(repl)
    if w_old > 0:
        new_len = max(6, int(round((f1 - f0) * w_new / w_old)))
    else:
        new_len = max(6, int(round(est.calculate_total_weight(repl) / 2.27)))
    log(f"weights old {w_old:.2f} new {w_new:.2f} -> {new_len} frames")
    log(f"source {total} frames ({total/FRAME_RATE:.2f}s); replace {f0}:{f1} "
        f"({(f1-f0)/FRAME_RATE:.2f}s) with {new_len} frames "
        f"({new_len/FRAME_RATE:.2f}s) for {repl!r}")

    target = np.concatenate(
        [src[:, :f0], np.repeat(np.asarray(mask_ids, dtype=np.int64)[:, None], new_len, 1),
         src[:, f1:]], axis=1)
    editable = np.zeros(target.shape[1], dtype=np.int8)
    editable[f0:f0 + new_len] = 1
    log(f"target {target.shape}, editable {int(editable.sum())} at {f0}:{f0+new_len}")
    if os.environ.get("VV_DRYRUN"):
        return 0

    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("VV_DTYPE", "bf16")]
    gen = build_generator(dt)
    gcfg = LLMGenerationConfig(num_step=int(os.environ.get("VV_STEPS", "32")),
                               guidance_scale=float(os.environ.get("VV_CFG", "2.0")))
    t0 = time.perf_counter()
    out = gen.generate_edit_masked(
        full_text=full_text,
        target_audio_tokens=torch.as_tensor(target),
        editable_audio_mask=torch.as_tensor(editable).bool(),
        language="zh",
        generation_config=gcfg,
        edit_context_frames=ctx,
    )
    log(f"generated {tuple(out.shape)} in {time.perf_counter()-t0:.0f}s")
    tk = out.numpy().astype(np.int32)
    np.save(out_path, tk)
    ch = (tk != target).any(axis=0)
    idx = np.flatnonzero(ch)
    log(f"saved {out_path}; changed frames {ch.sum()} range {idx[0]}:{idx[-1]+1}")
    log(f"frozen left/right: {not ch[:f0].any()} / {not ch[f0+new_len:].any()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
