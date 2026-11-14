#!/usr/bin/env python3
"""Export the DualCodec semantic frontend (mel + W2V-BERT 2.0) to ONNX.

viitorvoice/codec/encoder.py `_extract_semantic_repr` does:

    fe = SeamlessM4TFeatureExtractor(...)        # kaldi fbank 80 bins, stride 2
    out = Wav2Vec2BertModel(input_features=...)  # hidden_states[-2]
    feat = (out - mean) / std                    # emilia stats
    feat = avg_pool1d(feat.T, 2, 2)              # (1, 1024, T/4)  -> 25 fps

The old w2v_bert_2_0.onnx fed raw audio to the *model* (which wants mel
features) and was frozen at 1 s, so it was never usable.  This export bakes the
whole chain into one graph with a dynamic time axis:

    audio16k (1, N) f32  ->  semantic_repr (1, 1024, N/640) f32

stages: mel        -> check the torch fbank against transformers' numpy one
        full       -> check the whole frontend against the HF reference path
        export     -> write ONNX and check ORT against torch
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
OUT = os.path.expanduser("~/tmp/dl_vv/w2v_frontend.onnx")

FRAME_LEN, HOP, FFT, PREEMPH = 400, 160, 512, 0.97
MEL_FLOOR = 1.192092955078125e-07


def load_audio16k():
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    sr, d = wavfile.read(WAV)
    d = d.astype(np.float32) / 32768.0 if d.dtype == np.int16 else d.astype(np.float32)
    if d.ndim == 2:
        d = d.mean(1)
    if sr != 16000:
        g = np.gcd(int(sr), 16000)
        d = resample_poly(d, 16000 // g, int(sr) // g).astype(np.float32)
    return torch.from_numpy(d)


def mel_consts():
    from transformers.audio_utils import mel_filter_bank, window_function
    mf = mel_filter_bank(num_frequency_bins=257, num_mel_filters=80, min_frequency=20,
                         max_frequency=8000, sampling_rate=16000, norm=None,
                         mel_scale="kaldi", triangularize_in_mel_space=True)
    win = window_function(FRAME_LEN, "povey", periodic=False)
    return torch.from_numpy(mf).float(), torch.from_numpy(win).float()


def torch_fbank(audio, mel_filters, window):
    """(N,) float32 in [-1,1] -> (T, 80) log-mel, transformers' kaldi recipe."""
    x = audio.float() * 32768.0
    frames = x.unfold(0, FRAME_LEN, HOP)                      # (T, 400)
    frames = frames - frames.mean(-1, keepdim=True)
    y = torch.empty_like(frames)
    y[:, 0] = frames[:, 0] * (1.0 - PREEMPH)
    y[:, 1:] = frames[:, 1:] - PREEMPH * frames[:, :-1]
    y = y * window
    spec = torch.fft.rfft(y, n=FFT).abs() ** 2                # (T, 257)
    mel = spec @ mel_filters                                  # (T, 80)
    return torch.log(torch.clamp(mel, min=MEL_FLOOR))


class Frontend(torch.nn.Module):
    def __init__(self, w2v, mel_filters, window, mean, std):
        super().__init__()
        self.w2v = w2v
        self.register_buffer("mel_filters", mel_filters)
        self.register_buffer("window", window)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, audio):                                  # (1, N)
        mel = torch_fbank(audio[0], self.mel_filters, self.window)   # (T, 80)
        m = mel.mean(0, keepdim=True)
        v = mel.var(0, unbiased=True, keepdim=True)
        mel = (mel - m) / torch.sqrt(v + 1e-7)
        # transformers pads the (normalised) feature matrix to an even frame
        # count with zeros before the stride-2 concat; the pad frame is masked.
        t = mel.shape[0]
        if t % 2:   # the checkpoint pads with padding_value=1 (not 0)
            mel = torch.cat([mel, torch.ones(1, mel.shape[1], dtype=mel.dtype,
                                             device=mel.device)], 0)
        t2 = mel.shape[0] // 2
        feats = mel[None].reshape(1, t2, 160)
        mask = torch.zeros(1, t2, dtype=torch.long)
        mask[:, : t // 2] = 1
        h = self.w2v(input_features=feats, attention_mask=mask,
                     output_hidden_states=True).hidden_states[-2]
        h = (h - self.mean) / self.std
        return F.avg_pool1d(h.transpose(1, 2), 2, 2)                 # (1, 1024, T/4)


def stage_mel():
    audio = load_audio16k()
    mel_filters, window = mel_consts()
    from transformers import AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained(W2V_DIR)
    ref = fe(audio.numpy(), sampling_rate=16000, return_tensors="pt")
    mel = torch_fbank(audio, mel_filters, window)
    # transformers normalises per mel bin (ddof=1), pads to an even frame count
    # with zeros, then concatenates pairs of frames
    m = mel.mean(0, keepdim=True)
    v = mel.var(0, unbiased=True, keepdim=True)
    mn = (mel - m) / torch.sqrt(v + 1e-7)
    t = mn.shape[0]
    if t % 2:
        mn = torch.cat([mn, torch.ones(1, 80)], 0)
    a = mn.reshape(-1, 160)
    b = ref["input_features"][0]
    print("stride2 frames: ref", tuple(b.shape), "mine", tuple(a.shape))
    d = (a - b).abs()
    print(f"max|d| {d.max():.6f} mean|d| {d.mean():.8f}")
    print(f"cos {float((a.flatten()*b.flatten()).sum()/(a.norm()*b.norm())):.8f}")
    return 0


def build_model():
    from transformers import AutoModel
    t0 = time.perf_counter()
    w2v = AutoModel.from_pretrained(W2V_DIR, torch_dtype=torch.float32).eval()
    print(f"[w2v] loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    mean = torch.from_numpy(np.fromfile(STATS_MEAN, dtype=np.float32))
    std = torch.from_numpy(np.fromfile(STATS_STD, dtype=np.float32))
    mel_filters, window = mel_consts()
    return Frontend(w2v, mel_filters, window, mean, std).eval()


def hf_reference(audio):
    """Exactly what official_edit.stage_encode does."""
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(W2V_DIR)
    model = AutoModel.from_pretrained(W2V_DIR, torch_dtype=torch.float32).eval()
    inp = fe(audio.numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = model(input_features=inp["input_features"],
                    attention_mask=inp["attention_mask"], output_hidden_states=True)
    mean = torch.from_numpy(np.fromfile(STATS_MEAN, dtype=np.float32))
    std = torch.from_numpy(np.fromfile(STATS_STD, dtype=np.float32))
    feat = (out.hidden_states[-2] - mean) / std
    return F.avg_pool1d(feat.transpose(1, 2), 2, 2)


def stage_full():
    """One model load: HF path (feature extractor + model) vs our torch frontend."""
    import gc
    audio = load_audio16k()
    torch.set_grad_enabled(False)
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(W2V_DIR)
    w2v = AutoModel.from_pretrained(W2V_DIR, torch_dtype=torch.float32).eval()
    print("[w2v] loaded", flush=True)
    inp = fe(audio.numpy(), sampling_rate=16000, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = w2v(input_features=inp["input_features"],
                  attention_mask=inp["attention_mask"], output_hidden_states=True)
    print(f"[hf] {time.perf_counter()-t0:.1f}s  feats {tuple(inp['input_features'].shape)}",
          flush=True)
    mean = torch.from_numpy(np.fromfile(STATS_MEAN, dtype=np.float32))
    std = torch.from_numpy(np.fromfile(STATS_STD, dtype=np.float32))
    ref = F.avg_pool1d(((out.hidden_states[-2] - mean) / std).transpose(1, 2), 2, 2)
    del out
    gc.collect()
    print("ref", tuple(ref.shape), flush=True)

    mel_filters, window = mel_consts()
    model = Frontend(w2v, mel_filters, window, mean, std).eval()
    t0 = time.perf_counter()
    got = model(audio[None])
    print(f"[frontend] {time.perf_counter()-t0:.1f}s", flush=True)
    a = got.flatten().double()
    b = ref.flatten().double()
    c = float((a * b).sum() / (a.norm() * b.norm()))
    print(f"frontend vs HF: cos {c:.8f}  max|d| {(got-ref).abs().max():.6f}  "
          f"max|ref| {ref.abs().max():.2f}")
    return 0


def stage_export():
    audio = load_audio16k()
    torch.set_grad_enabled(False)
    model = build_model()
    dummy = audio[None][:, :48000]        # 3 s is enough to trace
    t0 = time.perf_counter()
    ep = torch.export.export(model, (dummy,), dynamic_shapes=({1: torch.export.Dim("n", min=8000)},))
    print(f"[export] traced in {time.perf_counter()-t0:.1f}s", flush=True)
    torch.onnx.export(ep, (dummy,), OUT, input_names=["audio16k"],
                      output_names=["semantic_repr"], dynamo=True)
    import onnx
    from onnx import numpy_helper
    m = onnx.load(OUT)
    for t in m.graph.initializer:
        arr = numpy_helper.to_array(t, base_dir=os.path.dirname(OUT))
        t.CopyFrom(numpy_helper.from_array(arr, name=t.name))
    onnx.save(m, OUT)
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
    s_ = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    for n in (48000, audio.shape[0]):
        x = audio[None][:, :n].numpy().astype(np.float32)
        t0 = time.perf_counter()
        y = s_.run(None, {"audio16k": x})[0]
        print(f"  ORT n={n} -> {y.shape} in {time.perf_counter()-t0:.2f}s", flush=True)
        if n == audio.shape[0]:
            ref = hf_reference(audio).numpy()
            print(f"  ORT vs HF cos "
                  f"{float((y.flatten()@ref.flatten())/(np.linalg.norm(y)*np.linalg.norm(ref))):.8f}")
    return 0


if __name__ == "__main__":
    torch.set_num_threads(6)
    stage = sys.argv[1] if len(sys.argv) > 1 else "mel"
    sys.exit({"mel": stage_mel, "full": stage_full, "export": stage_export}[stage]())
