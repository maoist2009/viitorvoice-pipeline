#!/usr/bin/env python3
"""Export the app-side semantic frontend: 8 s of 16 kHz audio -> semantic_repr.

Same maths as export_w2v_frontend.py (verified cos 1.00000000 against the HF
path) but with a fixed 8 s shape so torch.export does not have to solve dynamic
constraints on a 0.6 B model (that hangs / gets killed on the phone).

The app always feeds exactly 8 s (128000 samples, zero padded) plus the real
length, so the per-mel-bin normalisation stays correct for short clips:

    audio16k (1, 128000) f32, valid_len (1,) int64
      -> semantic_repr (1, 1024, 200) f32      # 200 frames = 8 s @ 25 Hz

stages: full (torch vs HF), export (ONNX + ORT vs HF)
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/viitor-voice-nar/local_models"
W2V_DIR = os.path.join(ROOT, "dualcodec/w2v-bert-2.0")
STATS_MEAN = os.path.join(ROOT, "dualcodec/dualcodec_ckpts/w2vbert2_mean_var_stats_emilia_mean.bin")
STATS_STD = os.path.join(ROOT, "dualcodec/dualcodec_ckpts/w2vbert2_mean_var_stats_emilia_std.bin")
WAV = "/data/data/com.termux/files/home/projects/voice_editor/viitor-voice-nar/assets/edit_prompt_zh.wav"
OUT = "/data/data/com.termux/files/usr/tmp/vv_nar/w2v_front8s.onnx"

FRAME_LEN, HOP, FFT, PREEMPH = 400, 160, 512, 0.97
MEL_FLOOR = 1.192092955078125e-07
N_SAMPLES = 128640                      # 8.04 s @ 16 kHz -> exactly 200 frames
N_MEL = 1 + (N_SAMPLES - FRAME_LEN) // HOP      # 802 (even, no pad row)
N_STRIDE = N_MEL // 2                           # 401 -> conv subsample -> 200
N_FRAMES = 200                                  # 8 s @ 25 Hz


def load_audio16k(seconds=8.0):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    sr, d = wavfile.read(WAV)
    d = d.astype(np.float32) / 32768.0 if d.dtype == np.int16 else d.astype(np.float32)
    if d.ndim == 2:
        d = d.mean(1)
    if sr != 16000:
        g = np.gcd(int(sr), 16000)
        d = resample_poly(d, 16000 // g, int(sr) // g).astype(np.float32)
    d = d[: int(seconds * 16000)]
    valid = len(d)
    pad = np.zeros(N_SAMPLES, dtype=np.float32)
    pad[:valid] = d
    return torch.from_numpy(pad), valid


def mel_consts():
    from transformers.audio_utils import mel_filter_bank, window_function
    mf = mel_filter_bank(num_frequency_bins=257, num_mel_filters=80, min_frequency=20,
                         max_frequency=8000, sampling_rate=16000, norm=None,
                         mel_scale="kaldi", triangularize_in_mel_space=True)
    win = window_function(FRAME_LEN, "povey", periodic=False)
    return torch.from_numpy(mf).float(), torch.from_numpy(win).float()


def torch_fbank(audio, mel_filters, window):
    x = audio.float() * 32768.0
    frames = x.unfold(0, FRAME_LEN, HOP)
    frames = frames - frames.mean(-1, keepdim=True)
    y = torch.empty_like(frames)
    y[:, 0] = frames[:, 0] * (1.0 - PREEMPH)
    y[:, 1:] = frames[:, 1:] - PREEMPH * frames[:, :-1]
    y = y * window
    spec = torch.fft.rfft(y, n=FFT).abs() ** 2
    mel = spec @ mel_filters
    return torch.log(torch.clamp(mel, min=MEL_FLOOR))


class Frontend8s(torch.nn.Module):
    def __init__(self, w2v, mel_filters, window, mean, std):
        super().__init__()
        self.w2v = w2v
        self.register_buffer("mel_filters", mel_filters)
        self.register_buffer("window", window)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("idx", torch.arange(N_MEL, dtype=torch.float32))
        self.register_buffer("sidx", torch.arange(N_STRIDE, dtype=torch.float32))

    def forward(self, audio, valid_len):
        mel = torch_fbank(audio[0], self.mel_filters, self.window)      # (802, 80)
        # per-mel-bin normalisation over the *real* frames only
        v = torch.floor((valid_len - FRAME_LEN) / HOP) + 1.0              # mel frames
        keep = (self.idx < v)[:, None].float()                          # (802, 1)
        s = (mel * keep).sum(0)
        mu = s / v
        var = (((mel - mu) ** 2) * keep).sum(0) / (v - 1)
        mel = (mel - mu) / torch.sqrt(var + 1e-7)
        feats = mel[None].reshape(1, N_STRIDE, 160)
        # The mask hides the padded tail from attention; the output length is
        # set by the conv adapter, not by the mask, so it stays static.
        amask = (self.sidx * 2.0 < v).long()
        h = self.w2v(input_features=feats, attention_mask=amask[None],
                     output_hidden_states=True).hidden_states[-2]
        h = (h - self.mean) / self.std
        return F.avg_pool1d(h.transpose(1, 2), 2, 2)                    # (1, 1024, 200)


def build_model(w2v):
    mean = torch.from_numpy(np.fromfile(STATS_MEAN, dtype=np.float32))
    std = torch.from_numpy(np.fromfile(STATS_STD, dtype=np.float32))
    mf, win = mel_consts()
    return Frontend8s(w2v, mf, win, mean, std).eval()


def hf_reference(w2v, audio_np):
    """What official_edit.stage_encode does for the same clip."""
    from transformers import AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained(W2V_DIR)
    inp = fe(audio_np, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = w2v(input_features=inp["input_features"],
                  attention_mask=inp["attention_mask"], output_hidden_states=True)
    mean = torch.from_numpy(np.fromfile(STATS_MEAN, dtype=np.float32))
    std = torch.from_numpy(np.fromfile(STATS_STD, dtype=np.float32))
    return F.avg_pool1d(((out.hidden_states[-2] - mean) / std).transpose(1, 2), 2, 2)


def stage_full():
    from transformers import AutoModel
    torch.set_grad_enabled(False)
    w2v = AutoModel.from_pretrained(W2V_DIR, torch_dtype=torch.float32).eval()
    audio, valid = load_audio16k()
    real = audio[:valid].numpy()
    ref = hf_reference(w2v, real)
    print("hf ref", tuple(ref.shape), flush=True)
    model = build_model(w2v)
    got = model(audio[None], torch.tensor([float(valid)]))
    a, b = got.flatten().double(), ref.flatten().double()
    print(f"frontend8s vs HF: cos {float((a*b).sum()/(a.norm()*b.norm())):.8f} "
          f"max|d| {(got-ref).abs().max():.6f}", flush=True)
    return 0


def stage_export():
    from transformers import AutoModel
    torch.set_grad_enabled(False)
    w2v = AutoModel.from_pretrained(W2V_DIR, torch_dtype=torch.float32).eval()
    audio, valid = load_audio16k()
    real = audio[:valid].numpy()
    model = build_model(w2v)
    dummy = (torch.zeros(1, N_SAMPLES), torch.tensor([float(valid)]))
    t0 = time.perf_counter()
    ep = torch.export.export(model, dummy)
    print(f"[export] traced {time.perf_counter()-t0:.1f}s", flush=True)
    torch.onnx.export(ep, dummy, OUT, input_names=["audio16k", "valid_len"],
                      output_names=["semantic_repr"], dynamo=True)
    import onnx
    m = onnx.load(OUT, load_external_data=False)   # weights stay in .onnx.data
    onnx.checker.check_model(OUT)
    print(f"[ok] {OUT} {os.path.getsize(OUT)/1e6:.0f} MB", flush=True)
    for i in m.graph.input:
        print("  IN ", i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
    for o in m.graph.output:
        print("  OUT", o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
    del model, m, ep
    import gc
    gc.collect()

    import onnxruntime as ort
    ref = hf_reference(w2v, real).numpy()
    del w2v
    gc.collect()
    s_ = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    y = s_.run(None, {"audio16k": audio[None].numpy().astype(np.float32),
                      "valid_len": np.array([valid], dtype=np.float32)})[0]
    print(f"  ORT -> {y.shape} in {time.perf_counter()-t0:.2f}s", flush=True)
    a, b = y.flatten().astype(np.float64), ref.flatten().astype(np.float64)
    print(f"  ORT vs HF cos {float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b))):.8f} "
          f"max|d| {np.abs(y-ref).max():.6f}")
    return 0


if __name__ == "__main__":
    torch.set_num_threads(6)
    stage = sys.argv[1] if len(sys.argv) > 1 else "full"
    sys.exit({"full": stage_full, "export": stage_export}[stage]())
