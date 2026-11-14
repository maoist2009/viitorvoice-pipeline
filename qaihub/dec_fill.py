#!/usr/bin/env python3
"""Pack generated tokens like DualCodecPackedDecoder (25 silence frames on both
sides), decode through the NPU chain and trim back to the real target."""
import os
import sys
import wave

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402

SIL = ("/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/"
       "viitor-voice-nar/local_models/assets/dualcodec_silence_2s.pt")


def main():
    src = sys.argv[1]
    out = sys.argv[2]
    tk = np.load(src)
    d = torch.load(SIL, map_location="cpu")
    sil = torch.as_tensor(d["tokens"] if isinstance(d, dict) else d).long().numpy().astype(np.int64)
    gap = sil[:, :25]
    packed = np.concatenate([gap, tk, gap], axis=1)
    if packed.shape[1] < P.FRAMES:
        need = P.FRAMES - packed.shape[1]
        reps = int(np.ceil(need / sil.shape[1]))
        packed = np.concatenate([packed, np.tile(sil, (1, reps))[:, :need]], axis=1)
    packed = packed[:, :P.FRAMES]
    print(f"[dec] {tk.shape[1]} target frames -> packed {packed.shape[1]}", flush=True)
    P.decoder_raw(packed, out)
    w = wave.open(out)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    lead = 25 * 960
    seg = a[lead:lead + tk.shape[1] * 960]
    pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16)
    with wave.open(out, "wb") as ww:
        ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(24000)
        ww.writeframes(pcm.tobytes())
    n = 2400
    env = [float(np.sqrt((seg[i:i+n]**2).mean())) for i in range(0, len(seg) - n, n)]
    print("[out] %.2fs rms %.4f" % (len(seg)/24000, float(np.sqrt((seg**2).mean()))), flush=True)
    print("[out] env", " ".join("%.2f" % x for x in env[:40]), flush=True)


if __name__ == "__main__":
    main()
