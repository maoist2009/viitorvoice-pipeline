#!/usr/bin/env python3
"""Official "Voice Edit" (text local edit) pipeline, run in-process on the phone.

This mirrors the shipping gRPC demo (encoder -> aligner -> local_edit plan ->
llm.generate_edit_masked -> decoder) but without the gRPC layer, and with CPU
stand-ins for the two classes that hard-require CUDA (DualCodec encoder) and the
missing `dualcodec`/`librosa`/`soundfile` python packages.

Stages are separate processes so the 2.2 GB w2v ONNX and the LLM never have to
be resident at the same time:

    edit encode   <wav> <tokens.npy>
    edit align    <wav> <original_text> <items.json>      [lang=zh]
    edit plan     <tokens.npy> <items.json> <original> <edited> <plan_dir>
    edit generate <plan_dir> <tokens_out.npy>             [context_frames]
    edit decode   <tokens.npy> <out.wav>
    edit all      <wav> <original_text> <edited_text> <out_prefix>
"""
import gc
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = "/data/data/com.termux/files/home/projects/voice_editor/viitor-voice-nar"
MODELS = ("/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/"
          "viitor-voice-nar/local_models")
SHIM = "/data/data/com.termux/files/usr/tmp/vv_shim"
QWEN_ASR_SRC = "/data/data/com.termux/files/home/tmp/asr/qwenasr_src"
WORK = "/data/data/com.termux/files/usr/tmp/vv_edit"

W2V_DIR = os.path.join(MODELS, "dualcodec/w2v-bert-2.0")
STATS_MEAN = os.path.join(MODELS, "dualcodec/dualcodec_ckpts/w2vbert2_mean_var_stats_emilia_mean.bin")
STATS_STD = os.path.join(MODELS, "dualcodec/dualcodec_ckpts/w2vbert2_mean_var_stats_emilia_std.bin")
ENC_ONNX = os.path.join(MODELS, "dualcodec/dualcodec_ckpts/dualcodec_encode_core_30s.onnx")
DEC_ONNX = os.path.join(MODELS, "dualcodec/dualcodec_ckpts/dualcodec_decoder.onnx")
ALIGNER = os.path.join(MODELS, "aligner/Qwen3-ForcedAligner-0.6B")
SILENCE = os.path.join(MODELS, "assets/dualcodec_silence_2s.pt")
LLM_DIR = os.path.join(MODELS, "llm/0p6_emotion")

SR = 24000
W2V_SR = 16000
FRAME_RATE = 25
HOP = 960


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ stage: encode
def stage_encode(wav_path, out_npy):
    import onnxruntime as ort
    from scipy.signal import resample_poly

    from scipy.io import wavfile
    sr, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    log(f"audio {len(data)/sr:.2f}s @{sr} Hz")
    if sr != SR:
        g = np.gcd(int(sr), SR)
        data = resample_poly(data, SR // g, int(sr) // g).astype(np.float32)
        sr = SR

    a16 = resample_poly(data, W2V_SR, SR).astype(np.float32)
    # The semantic frontend is Wav2Vec2BertModel: it wants mel `input_features`
    # (80 bins, stride 2 -> 160), exactly like viitorvoice/codec/encoder.py
    # `_extract_semantic_repr`.  The old w2v_bert_2_0.onnx feeds it raw audio,
    # which is wrong -- so run the HF model directly here.
    import torch
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModel
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    log(f"w2v ({W2V_SR} Hz, {len(a16)/W2V_SR:.2f}s)")
    fe = AutoFeatureExtractor.from_pretrained(W2V_DIR)
    model = AutoModel.from_pretrained(W2V_DIR, torch_dtype=torch.float32).eval()
    inputs = fe(a16, sampling_rate=W2V_SR, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_features=inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True)
    log(f"w2v forward {time.perf_counter()-t0:.1f}s, "
        f"{len(out.hidden_states)} hidden states {tuple(out.hidden_states[-2].shape)}")
    mean = np.fromfile(STATS_MEAN, dtype=np.float32)
    std = np.fromfile(STATS_STD, dtype=np.float32)
    feat = (out.hidden_states[-2] - torch.from_numpy(mean)) / torch.from_numpy(std)
    feat = F.avg_pool1d(feat.transpose(1, 2), 2, 2)
    feat = feat.numpy().astype(np.float32)
    log(f"semantic_repr {feat.shape}")
    del model, out
    gc.collect()

    s_enc = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])
    audio_in = data[None, None, :].astype(np.float32)
    outs = s_enc.run(None, {"audio": audio_in, "semantic_repr": feat.astype(np.float32)})
    sem, aco = outs[0], outs[1]
    tokens = np.concatenate([sem, aco], axis=1)[0].astype(np.int32)
    log(f"tokens {tokens.shape} sem[{sem.min()},{sem.max()}] aco[{aco.min()},{aco.max()}]")
    np.save(out_npy, tokens)
    log(f"saved {out_npy}")


# ------------------------------------------------------------------- stage: align
def stage_align(wav_path, text, out_json, language="zh", granularity="char"):
    for p in (QWEN_ASR_SRC, SHIM, IMPL):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(6)
    from viitorvoice.local_edit.alignment import Qwen3ForcedAlignerWrapper

    # `accelerate` is not installed here, and the wrapper unconditionally passes
    # device_map=... to from_pretrained.  CPU needs no device_map at all.
    Qwen3ForcedAlignerWrapper._normalize_device_map = staticmethod(lambda device: None)

    log(f"loading aligner {ALIGNER}")
    aligner = Qwen3ForcedAlignerWrapper(model_path=ALIGNER, device="cpu",
                                        default_language=language, dtype="fp32")
    t0 = time.perf_counter()
    items = aligner.align(audio=wav_path, text=text, language=language,
                          granularity=granularity)
    log(f"aligned {len(items)} items in {time.perf_counter()-t0:.1f}s")
    rows = [it.to_dict() for it in items]
    for row in rows[:8]:
        log(f"  {row['index']:3d} {row['text']!r} {row['start_time']:.3f}-{row['end_time']:.3f} "
            f"chars {row['start_char']}:{row['end_char']}")
    json.dump(rows, open(out_json, "w"), ensure_ascii=False, indent=1)
    log(f"saved {out_json}")


# -------------------------------------------------------------------- stage: plan
def _word_diff_needed(language):
    return (language or "").lower() in {"en"}


def _expanded_changed_spans(original, edited):
    """servicer._changed_text_spans for the char (CJK) granularity."""
    import difflib
    opcodes = difflib.SequenceMatcher(a=original, b=edited, autojunk=False).get_opcodes()
    spans = []
    for tag, start, end, estart, eend in opcodes:
        if tag == "equal":
            continue
        if start == end:
            if start > 0:
                spans.append((start - 1, start, estart - 1, eend))
            elif original:
                spans.append((0, 1, estart, eend + 1))
            else:
                raise SystemExit("empty original text")
        elif estart == eend:
            if start > 0:
                spans.append((start - 1, end, estart - 1, eend))
            elif end < len(original):
                spans.append((start, end + 1, estart, eend + 1))
            else:
                raise SystemExit("deleting the whole text is not supported")
        else:
            spans.append((start, end, estart, eend))
    if not spans:
        raise SystemExit("edited text is identical to the original")

    ordered = sorted(spans, key=lambda s: (s[0], s[1]))
    merged = []
    for start, end, estart, eend in ordered:
        if not merged:
            merged.append((start, end, estart, eend))
            continue
        sep = original[merged[-1][1]:start]
        if start > merged[-1][1] and any(c.isalnum() for c in sep):
            merged.append((start, end, estart, eend))
            continue
        ps, pe, pes, pee = merged[-1]
        merged[-1] = (ps, max(pe, end), min(pes, estart), max(pee, eend))
    return opcodes, [(s, e) for s, e, _, _ in merged]


def _indices_for_char_span(items, start, end):
    indexed = [(it.index, it.start_char, it.end_char) for it in items
               if it.start_char is not None and it.end_char is not None
               and it.end_char > it.start_char]
    if not indexed:
        raise SystemExit("alignment items have no char spans")
    picked = [i for i, s, e in indexed if e > start and s < end]
    if picked:
        return sorted(set(picked))
    after = [(s, i) for i, s, _ in indexed if s >= end]
    if after:
        return [min(after)[1]]
    before = [(e, i) for i, _, e in indexed if e <= start]
    if before:
        return [max(before)[1]]
    raise SystemExit(f"cannot map char span {start}:{end}")


def _char_span_for_indices(items, indices):
    wanted = set(indices)
    sel = [it for it in items if it.index in wanted and it.start_char is not None
           and it.end_char is not None and it.end_char > it.start_char]
    if not sel:
        raise SystemExit("indices have no char spans")
    return min(it.start_char for it in sel), max(it.end_char for it in sel)


def _edited_span_for_original_span(opcodes, start, end):
    ranges = []
    for tag, os_, oe, es, ee in opcodes:
        if tag == "insert":
            if start <= os_ <= end or (os_ == end and start == end):
                ranges.append((es, ee))
        elif tag == "replace":
            if oe > start and os_ < end:
                ranges.append((es, ee))
        elif tag == "delete":
            if oe > start and os_ < end:
                ranges.append((es, es))
        elif os_ >= end:
            break
    if not ranges:
        return start, end
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def stage_plan(tokens_npy, items_json, original, edited, plan_dir, language="zh",
               expand_mask_ratio=1.5):
    for p in (IMPL,):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    from viitorvoice.local_edit import AlignmentItem, build_multi_local_edit_token_plan

    tokens = np.load(tokens_npy)
    rows = json.load(open(items_json))
    items = [AlignmentItem(**{k: r[k] for k in
                              ("index", "text", "start_time", "end_time",
                               "start_char", "end_char", "kind")}) for r in rows]

    opcodes, spans = _expanded_changed_spans(original, edited)
    groups = []
    for start, end in spans:
        idx = _indices_for_char_span(items, start, end)
        if groups and (set(groups[-1]) & set(idx) or idx[0] <= groups[-1][-1] + 1):
            groups[-1] = sorted(set(groups[-1]) | set(idx))
        else:
            groups.append(idx)
    selections, replacements = [], []
    for idx in groups:
        cs, ce = _char_span_for_indices(items, idx)
        es, ee = _edited_span_for_original_span(opcodes, cs, ce)
        runs, s = [], idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
                continue
            runs.append(f"{s}-{prev}" if s != prev else str(s))
            s = prev = i
        runs.append(f"{s}-{prev}" if s != prev else str(s))
        selections.append(",".join(runs))
        replacements.append(edited[es:ee])
        log(f"edit chars[{cs}:{ce}] -> indices {idx} selection {selections[-1]!r} "
            f"replacement {replacements[-1]!r}")

    os.makedirs(plan_dir, exist_ok=True)
    plan = build_multi_local_edit_token_plan(
        source_tokens=torch.as_tensor(tokens, dtype=torch.long),
        alignment_items=items,
        original_text=original,
        selection=";".join(selections),
        replacement_text=replacements,
        audio_duration=tokens.shape[1] / FRAME_RATE,
        padding_ms=0.0,
        expand_mask_ratio=float(expand_mask_ratio),
        frame_rate=FRAME_RATE,
        length_mode="auto",
        length_scale=1.0,
        min_edit_frames=6,
        reference_context_frames=300,
    )
    np.save(os.path.join(plan_dir, "target.npy"), plan.target_tokens.numpy().astype(np.int32))
    np.save(os.path.join(plan_dir, "editable.npy"),
            plan.editable_audio_mask.numpy().astype(np.int8))
    masks = np.flatnonzero(plan.editable_audio_mask.numpy())
    meta = {
        "source_frames": int(plan.source_tokens.shape[-1]),
        "target_frames": int(plan.target_tokens.shape[-1]),
        "editable_frames": int(plan.editable_audio_mask.sum()),
        "editable_range": [int(masks[0]), int(masks[-1])] if masks.size else None,
        "replacement_frames": [int(x) for x in plan.replacement_frames],
        "edited_full_text": plan.span.edited_full_text,
        "selection": ";".join(selections),
        "replacements": replacements,
        "language": language,
    }
    json.dump(meta, open(os.path.join(plan_dir, "meta.json"), "w"),
              ensure_ascii=False, indent=1)
    log("plan: " + json.dumps(meta, ensure_ascii=False))


# ---------------------------------------------------------------- stage: generate
def stage_generate(plan_dir, out_npy, dtype="fp32", context_frames=None, steps=32,
                   guidance=2.0):
    sys.path.insert(0, "/data/data/com.termux/files/home/projects/dreamlite/salvage/dl_vv")
    import torch
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from viitorvoice.llm.generation import LLMGenerationConfig
    from torch_generate import build_generator

    meta = json.load(open(os.path.join(plan_dir, "meta.json")))
    target = torch.as_tensor(np.load(os.path.join(plan_dir, "target.npy")), dtype=torch.long)
    editable = torch.as_tensor(np.load(os.path.join(plan_dir, "editable.npy"))).bool()
    log(f"target {tuple(target.shape)} editable {int(editable.sum())} "
        f"full_text {meta['edited_full_text']!r}")

    dt = torch.float32 if dtype == "fp32" else torch.bfloat16
    gen = build_generator(dt)
    cfg = LLMGenerationConfig(num_step=int(steps), guidance_scale=float(guidance))
    t0 = time.perf_counter()
    out = gen.generate_edit_masked(
        full_text=meta["edited_full_text"],
        target_audio_tokens=target,
        editable_audio_mask=editable,
        language=meta.get("language") or "zh",
        generation_config=cfg,
        edit_context_frames=context_frames,
    )
    log(f"generated {tuple(out.shape)} in {time.perf_counter()-t0:.0f}s")
    tokens = out.numpy().astype(np.int32)
    np.save(out_npy, tokens)
    changed = int((tokens != np.load(os.path.join(plan_dir, "target.npy"))).any(axis=0).sum())
    log(f"saved {out_npy}; frames changed vs plan target {changed}")
    # splice check: frames outside the editable window must be untouched
    if context_frames is None and tokens.shape[1] == int(meta["target_frames"]):
        lo, hi = meta["editable_range"][0], meta["editable_range"][1] + 1
        same = bool((tokens[:, :lo] == np.load(os.path.join(plan_dir, "target.npy"))[:, :lo]).all())
        same2 = bool((tokens[:, hi:] == np.load(os.path.join(plan_dir, "target.npy"))[:, hi:]).all())
        log(f"context frames frozen: left {same} right {same2}")


# -------------------------------------------------------------------- stage: decode
def stage_decode(tokens_npy, out_wav):
    import onnxruntime as ort
    import wave
    import torch

    tokens = np.load(tokens_npy)
    sil = torch.load(SILENCE, map_location="cpu")
    sil = torch.as_tensor(sil["tokens"] if isinstance(sil, dict) else sil).long().numpy()
    gap = sil[:, :25]
    packed = np.concatenate([gap, tokens, gap], axis=1)[None]
    log(f"decode {tokens.shape} -> packed {packed.shape}")
    sess = ort.InferenceSession(DEC_ONNX, providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    audio = sess.run(None, {"semantic_codes": packed[:, :1].astype(np.int64),
                            "acoustic_codes": packed[:, 1:].astype(np.int64)})[0].reshape(-1)
    log(f"decoded in {time.perf_counter()-t0:.1f}s, {len(audio)/SR:.2f}s")
    seg = audio[25 * HOP: (25 + tokens.shape[1]) * HOP - 4]
    pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    n = 2400
    env = [float(np.sqrt((seg[i:i+n]**2).mean())) for i in range(0, len(seg) - n, n)]
    log("wav %.2fs rms %.4f" % (len(seg) / SR, float(np.sqrt((seg ** 2).mean()))))
    log("env " + " ".join("%.2f" % x for x in env[:40]))
    log(f"saved {out_wav}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    os.makedirs(WORK, exist_ok=True)
    cmd = sys.argv[1]
    a = sys.argv[2:]
    if cmd == "encode":
        stage_encode(a[0], a[1] if len(a) > 1 else os.path.join(WORK, "src_tokens.npy"))
    elif cmd == "align":
        stage_align(a[0], a[1], a[2] if len(a) > 2 else os.path.join(WORK, "items.json"))
    elif cmd == "plan":
        stage_plan(a[0], a[1], a[2], a[3], a[4] if len(a) > 4 else os.path.join(WORK, "plan"))
    elif cmd == "generate":
        ctx = int(a[2]) if len(a) > 2 else None
        stage_generate(a[0], a[1], os.environ.get("VV_DTYPE", "fp32"), ctx,
                       int(os.environ.get("VV_STEPS", "32")),
                       float(os.environ.get("VV_CFG", "2.0")))
    elif cmd == "decode":
        stage_decode(a[0], a[1])
    elif cmd == "all":
        wav, original, edited, prefix = a[:4]
        stage_encode(wav, f"{prefix}_src_tokens.npy")
        stage_align(wav, original, f"{prefix}_items.json")
        stage_plan(f"{prefix}_src_tokens.npy", f"{prefix}_items.json", original, edited,
                   f"{prefix}_plan")
        stage_decode(f"{prefix}_src_tokens.npy", f"{prefix}_before.wav")
        stage_generate(f"{prefix}_plan", f"{prefix}_tokens.npy")
        stage_decode(f"{prefix}_tokens.npy", f"{prefix}_after.wav")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
