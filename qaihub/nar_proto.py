#!/usr/bin/env python3
"""Full ViiTorVoice-NAR pipeline prototype on the phone NPU (no ref audio).

text -> NAR masked-diffusion (llm.bin + head.bin on the HTP) -> codes -> the
5-graph decoder chain -> wav.

This is the reference implementation for the C++ port: every host-side step
follows viitorvoice/llm/generation.py exactly (32 steps, t_shift 0.1, layer
penalty 5.0, position temperature 5.0, no CFG), with the graphs swapped for the
compiled context binaries.

usage: nar_proto.py <text> [out.wav] [frames]

Env overrides: VV_STEPS (default 32), VV_TEMP (0 disables gumbel sampling).
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/viitor-voice-nar"
IMPL = "/data/data/com.termux/files/home/projects/voice_editor/viitor-voice-nar"
sys.path.insert(0, IMPL)
from viitorvoice.llm.text_utils import (  # noqa: E402
    RuleDurationEstimator, combine_text, insert_text_pause_anchors,
    prepare_text_for_tokenizer, tokenize_with_special_tokens)

MODEL = os.path.join(ROOT, "local_models/llm/0p6_emotion")
CKPT = os.path.join(MODEL, "model.safetensors")

QNN = os.path.expanduser("~/qairt/2.51.40")
CL = f"{QNN}/lib/aarch64-android"
BN = f"{QNN}/bin/aarch64-android"
SK = f"{QNN}/lib/hexagon-v79/unsigned"

LLM_BIN = os.environ.get("VV_LLM", os.path.join(HERE, "fixio/llm.bin"))
HEAD_BIN = os.path.join(HERE, "head.bin")
DEC = os.path.join(HERE, "dec_split")

SEQ = 256
N_CB = 12
MASK_IDS = [16386] + [1026] * 11
PAUSE_IDS = [16384] + [1024] * 11
SEP_IDS = [16385] + [1025] * 11
VOCAB_SEM = 16387
VOCAB_ACO = 1027
STEPS = int(os.environ.get("VV_STEPS", "32"))
TEMP = float(os.environ.get("VV_TEMP", "5.0"))
T_SHIFT = 0.1
LAYER_PENALTY = 5.0

CH, HALO = 47999, 48
FRAMES = 200          # the compiled decoder window is exactly 200 frames

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = f"/vendor/lib64:{CL}"
env["ADSP_LIBRARY_PATH"] = SK

WORK = "/data/data/com.termux/files/usr/tmp/vv_nar"
os.makedirs(WORK, exist_ok=True)


# ---------------------------------------------------------------- qnn helpers
def net_run(bin_path, inputs, out_dir, out_bytes):
    """One inference; inputs is a list of raw file paths in graph order.

    qnn-net-run drops the output tensors in `<out_dir>/Result_0/`; the caller
    sees them at the top level, and each call cleans its directory first so
    repeated calls never read a stale tensor.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.startswith("Result_"):
            import shutil
            shutil.rmtree(os.path.join(out_dir, name), ignore_errors=True)
    lst = os.path.join(out_dir, "list.txt")
    with open(lst, "w") as f:
        f.write(" ".join(inputs) + "\n")
    cmd = [f"{BN}/qnn-net-run", "--backend", f"{CL}/libQnnHtp.so",
           "--retrieve_context", bin_path, "--input_list", lst,
           "--output_dir", out_dir, "--use_native_input_files"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"qnn-net-run failed on {bin_path}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    dt = (time.perf_counter() - t0) * 1000
    res = os.path.join(out_dir, "Result_0")
    have = sorted(f for f in os.listdir(res) if f.endswith(".raw"))
    outs = []
    for i, (name, nbytes) in enumerate(out_bytes):
        path = os.path.join(res, name)
        if not os.path.exists(path):
            # single-output graphs keep their ONNX tensor name
            if len(have) == len(out_bytes):
                path = os.path.join(res, have[i])
            elif len(have) == 1:
                path = os.path.join(res, have[0])
            else:
                raise SystemExit(f"missing output {name} in {have}")
        outs.append(np.fromfile(path, dtype=np.float32))
        os.remove(path)
    return outs, dt


# ------------------------------------------------------------------ embeddings
class Tables:
    def __init__(self):
        self.tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(MODEL, "tokenizer.json"))
        self.f = safe_open(CKPT, framework="np")
        self.sem = self.f.get_tensor("semantic_embedding.weight")
        self.aco = self.f.get_tensor("acoustic_embedding.weight")
        self.offs = self.f.get_tensor("acoustic_codebook_offsets").astype(np.int64)
        self.text = self.f.get_slice("llm.embed_tokens.weight")
        self._rows = {}
        print(f"[emb] semantic {self.sem.shape} acoustic {self.aco.shape} "
              f"offsets {self.offs.tolist()}", flush=True)

    def text_rows(self, ids):
        """Gather embedding rows one by one (safetensors slicing, cached)."""
        out = np.empty((len(ids), 1024), dtype=np.float32)
        for i, tid in enumerate(ids):
            tid = int(tid)
            row = self._rows.get(tid)
            if row is None:
                row = self.text[tid:tid + 1][0].astype(np.float32)
                self._rows[tid] = row
            out[i] = row
        return out


def build_embeds(t, ids, audio_mask):
    """ids [12, S] int64, audio_mask [S] bool -> [1, S, 1024] float32."""
    text_ids = ids[0].astype(np.int64)
    text_e = t.text_rows(text_ids.tolist())
    sem_ids = np.where(audio_mask, ids[0], 0).astype(np.int64)
    sem_e = t.sem[sem_ids]
    aco_ids = np.where(audio_mask[None, :], ids[1:], 0).astype(np.int64)
    aco_e = np.zeros((ids.shape[1], 1024), dtype=np.float32)
    for c in range(11):
        aco_e += t.aco[aco_ids[c] + t.offs[c]]
    audio_e = sem_e + aco_e
    out = np.where(audio_mask[:, None], audio_e, text_e).astype(np.float32)
    return out[None]


# ---------------------------------------------------------------------- prompt
def estimate_frames(text, lang="zh"):
    """generation.py `_estimate_target_tokens` with the default ref clause
    ("Nice to meet you.", 1 s of audio at 25 fps)."""
    return max(1, int(RuleDurationEstimator().estimate_duration(
        text, "Nice to meet you.", 25)))


def build_prompt(t, text, lang, frames):
    style = (f"<|lang_start|>{lang}<|lang_end|>"
             f"<|instruct_start|>None<|instruct_end|>")
    style_ids = tokenize_with_special_tokens(style, t.tok).tolist()[0]
    full = combine_text(text=text, ref_text=None)
    full = prepare_text_for_tokenizer(full, t.tok, lang)
    full = insert_text_pause_anchors(full)
    text_ids = tokenize_with_special_tokens(f"<|text_start|>{full}<|text_end|>",
                                            t.tok).tolist()[0]
    cond = style_ids + text_ids
    total = len(cond) + frames
    if total > SEQ:
        raise SystemExit(f"prompt too long: {len(cond)} cond + {frames} target = {total} > {SEQ}")
    ids = np.zeros((N_CB, SEQ), dtype=np.int64)
    ids[:, :] = np.array(MASK_IDS, dtype=np.int64)[:, None]
    ids[:, :len(cond)] = np.repeat(np.array(cond, dtype=np.int64)[None, :], N_CB, axis=0)
    audio_mask = np.zeros(SEQ, dtype=bool)
    audio_mask[len(cond):total] = True
    attn = np.zeros((1, 1, SEQ, SEQ), dtype=np.float32)
    attn[0, 0, :total, :total] = 1.0
    idx = np.arange(total, SEQ)
    attn[0, 0, idx, idx] = 1.0          # keep padded rows finite
    return ids, audio_mask, attn, len(cond), total


# ---------------------------------------------------------------------- decode
def decoder(tokens, out_wav):
    """tokens [12, T] -> wav. Structural frames are dropped like the reference
    pipeline does (grpc_server.common.strip_structural_audio_frames), then the
    sequence is fitted to the compiled 200-frame window."""
    P = np.array(PAUSE_IDS, dtype=np.int64)[:, None]
    S = np.array(SEP_IDS, dtype=np.int64)[:, None]
    keep = ~(((tokens == P).all(0)) | ((tokens == S).all(0)))
    tk = tokens[:, keep]
    n = tk.shape[1]
    print(f"[dec] {n} speech frames of {tokens.shape[1]} (structural {int((~keep).sum())})", flush=True)
    if n == 0:
        raise SystemExit("no speech frames")
    keep_frames = n
    if n > FRAMES:
        tk, n = tk[:, :FRAMES], FRAMES
    elif n < FRAMES:
        tk = np.concatenate([tk, np.repeat(tk[:, -1:], FRAMES - n, axis=1)], axis=1)
        n = FRAMES
    _decode_frames(tk, out_wav, keep_frames=keep_frames)


def _decode_frames(tk, out_wav, keep_frames=None):
    """Exactly FRAMES columns of clean codes -> wav through the 5-graph chain."""
    tk = tk.copy()
    tk[0] = np.minimum(tk[0], 16383)       # p1's semantic table is 16384 rows
    tk[1:] = np.minimum(tk[1:], 1023)      # and the acoustic tables are 1024 rows
    sem = tk[0].astype(np.int32).reshape(1, 1, FRAMES)
    aco = tk[1:].astype(np.int32)[None]
    sem.tofile(os.path.join(WORK, "sem.raw"))
    aco.tofile(os.path.join(WORK, "aco.raw"))
    t0 = time.perf_counter()
    o, _ = net_run(os.path.join(DEC, "p1.bin"),
                   [os.path.join(WORK, "sem.raw"), os.path.join(WORK, "aco.raw")],
                   os.path.join(WORK, "d1"), [("output_0.raw", 768 * 1600 * 4)])
    x = o[0].reshape(1, 768, 1600)
    for i, (nm, elems) in enumerate([("p2", 384 * 9600), ("p3", 192 * 47999)], start=2):
        x.astype(np.float32).tofile(os.path.join(WORK, f"in{i}.raw"))
        o, _ = net_run(os.path.join(DEC, "a16", f"{nm}.bin"),
                       [os.path.join(WORK, f"in{i}.raw")], os.path.join(WORK, f"d{i}"),
                       [(f"output_{i-1}.raw", elems * 4)])
        x = o[0]
    x.tofile(os.path.join(WORK, "in4.raw"))
    o, _ = net_run(os.path.join(DEC, "a16", "p4.bin"), [os.path.join(WORK, "in4.raw")],
                   os.path.join(WORK, "d4"), [("output_3.raw", 96 * 191996 * 4)])
    c3 = o[0].reshape(1, 96, 191996)
    p5 = os.path.join(DEC, "a16", "p5chunk48.bin")
    total = c3.shape[2]
    chunks = []
    for i in range(4):
        start = i * CH - HALO
        s0, e0 = max(start, 0), min(start + CH + 2 * HALO, total)
        buf = np.zeros((1, 96, CH + 2 * HALO), dtype=np.float32)
        buf[:, :, s0 - start:e0 - start] = c3[:, :, s0:e0]
        buf.tofile(os.path.join(WORK, f"p5_{i}.raw"))
        o, _ = net_run(p5, [os.path.join(WORK, f"p5_{i}.raw")],
                       os.path.join(WORK, f"d5_{i}"), [("audio.raw", (CH + 2 * HALO) * 4)])
        chunks.append(o[0][HALO:HALO + CH])
    audio = np.concatenate(chunks)
    if keep_frames:
        audio = audio[:keep_frames * 960]      # 24000 Hz / 25 fps
    print(f"[dec] {time.perf_counter()-t0:.1f}s  {len(audio)} samples", flush=True)
    pcm = (np.clip(audio, -1, 1) * 32767.0).astype(np.int16)
    import wave
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())


decoder_raw = _decode_frames


# ------------------------------------------------------------------------ main
def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "你好，世界。今天我们测试一下这个语音合成系统。"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "nar_out.wav")
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if frames <= 0:
        frames = estimate_frames(text)
        print(f"[prompt] estimated target {frames} frames ({frames/25:.2f}s)", flush=True)

    t = Tables()
    ids, audio_mask, attn, cond_len, total = build_prompt(t, text, "zh", frames)
    print(f"[prompt] cond {cond_len} + target {frames} = {total}, steps {STEPS}", flush=True)

    rng = np.random.default_rng(0)
    tokens = np.array(MASK_IDS, dtype=np.int64)[:, None].repeat(frames, axis=1)
    layer_ids = np.arange(N_CB)[:, None].astype(np.float32)

    u = np.linspace(0.0, 1.0, STEPS + 1)
    tsteps = T_SHIFT * u / (1.0 + (T_SHIFT - 1.0) * u)
    total_mask = frames * N_CB
    remaining = total_mask
    schedule = []
    for s in range(STEPS):
        if s == STEPS - 1:
            k = remaining
        else:
            k = min(math.ceil(total_mask * (tsteps[s + 1] - tsteps[s])), remaining)
        schedule.append(int(k))
        remaining -= int(k)

    attn.tofile(os.path.join(WORK, "attn.raw"))
    t_all = time.perf_counter()
    llm_ms = head_ms = 0.0
    for step in range(STEPS):
        ids[:, cond_len:total] = tokens
        emb = build_embeds(t, ids, audio_mask)
        emb.tofile(os.path.join(WORK, "emb.raw"))
        (hid,), ms = net_run(LLM_BIN, [os.path.join(WORK, "emb.raw"),
                                      os.path.join(WORK, "attn.raw")],
                             os.path.join(WORK, "llm"), [("output_0.raw", SEQ * 1024 * 4)])
        llm_ms += ms
        hid.astype(np.float32).tofile(os.path.join(WORK, "hid.raw"))
        lg, ms = net_run(HEAD_BIN, [os.path.join(WORK, "hid.raw")],
                         os.path.join(WORK, "head"),
                         [("output_0.raw", SEQ * VOCAB_SEM * 4),
                          ("output_1.raw", SEQ * 11 * VOCAB_ACO * 4)])
        head_ms += ms
        sem = lg[0].reshape(SEQ, VOCAB_SEM)[cond_len:total].astype(np.float64)
        aco = lg[1].reshape(SEQ, 11, VOCAB_ACO)[cond_len:total].astype(np.float64)

        def logp(x):                       # log_softmax over last dim
            m = x.max(axis=-1, keepdims=True)
            e = np.exp(x - m)
            return (x - m) - np.log(e.sum(axis=-1, keepdims=True))

        sem_lp = logp(sem)
        aco_lp = logp(aco)
        sem_lp[:, MASK_IDS[0]] = -np.inf
        sem_lp[:, SEP_IDS[0]] = -np.inf
        aco_lp[:, :, MASK_IDS[1]] = -np.inf
        aco_lp[:, :, SEP_IDS[1]] = -np.inf
        pred = np.empty((N_CB, frames), dtype=np.int64)
        pred[0] = sem_lp.argmax(-1)
        scores = np.empty((N_CB, frames), dtype=np.float32)
        scores[0] = sem_lp.max(-1)
        for c in range(11):
            pred[c + 1] = aco_lp[:, c].argmax(-1)
            scores[c + 1] = aco_lp[:, c].max(-1)

        scores = scores - layer_ids * LAYER_PENALTY
        if TEMP > 0:
            g = -np.log(-np.log(rng.random(scores.shape) + 1e-10) + 1e-10)
            scores = scores / TEMP + g
        scores[tokens != np.array(MASK_IDS, dtype=np.int64)[:, None]] = -np.inf
        k = schedule[step]
        if k > 0:
            flat = scores.reshape(-1)
            top = np.argpartition(-flat, k - 1)[:k]
            fw = tokens.reshape(-1)
            fw[top] = pred.reshape(-1)[top]
            tokens = fw.reshape(N_CB, frames)
            # a predicted pause frame collapses all 12 codebooks to pause ids
            hits = top[top < frames]
            if hits.size:
                p = pred[0, hits]
                frames_pause = hits[p == PAUSE_IDS[0]]
                if frames_pause.size:
                    tokens[:, frames_pause] = np.array(PAUSE_IDS, dtype=np.int64)[:, None]
        filled = int((tokens != np.array(MASK_IDS, dtype=np.int64)[:, None]).sum())
        print(f"[step {step:2d}] fill {k:5d}  filled {filled}/{total_mask}", flush=True)

    print(f"[nar] {time.perf_counter()-t_all:.1f}s total, "
          f"llm {llm_ms/1000:.1f}s head {head_ms/1000:.1f}s", flush=True)
    np.save(os.path.join(WORK, "tokens.npy"), tokens)
    if os.environ.get("VV_NODEC"):
        return
    decoder(tokens, out)
    print("[done]", out, os.path.getsize(out), "B", flush=True)


if __name__ == "__main__":
    main()
