#!/usr/bin/env python3
"""Perceptual-ish quality metrics for the split decoder chain output.

usage: metrics.py <got.f32> <ref.npy>

Reports overall SNR, per-band SNR (speech energy is concentrated below 4 kHz,
so a broadband number understates intelligibility), and a log-mel distance.
"""
import sys

import numpy as np

got = np.load(sys.argv[1]).reshape(-1).astype(np.float32) if sys.argv[1].endswith(".npy") \
    else np.fromfile(sys.argv[1], dtype=np.float32)
ref = np.load(sys.argv[2]).reshape(-1).astype(np.float32)
n = min(got.size, ref.size)
got, ref = got[:n], ref[:n]


def snr(g, r):
    num = float(((g - r) ** 2).sum())
    den = float((r ** 2).sum())
    return 10 * np.log10(den / num)


print(f"samples {n}  ({n / 24000:.2f} s @24k)")
print(f"overall SNR   {snr(got, ref):6.2f} dB   corr {np.corrcoef(got, ref)[0, 1]:.4f}")
print(f"ref rms {np.sqrt((ref ** 2).mean()):.4f}   got rms {np.sqrt((got ** 2).mean()):.4f}")

# band SNR via FFT masking
N = 1 << 14
win = np.hanning(N).astype(np.float32)
step = N // 2
bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000)]
acc = {b: [0.0, 0.0] for b in bands}
freqs = np.fft.rfftfreq(N, 1 / 24000)
for s in range(0, n - N, step):
    a = np.fft.rfft(ref[s:s + N] * win)
    b = np.fft.rfft(got[s:s + N] * win)
    for band in bands:
        m = (freqs >= band[0]) & (freqs < band[1])
        acc[band][0] += float((np.abs(b[m] - a[m]) ** 2).sum())
        acc[band][1] += float((np.abs(a[m]) ** 2).sum())
print("\nband SNR (energy-weighted):")
for band, (e, r) in acc.items():
    if r <= 0:
        continue
    print(f"  {band[0]:>5}-{band[1]:<5} Hz  {10 * np.log10(r / e) if e > 0 else 99:6.2f} dB"
          f"   ({100 * r / sum(v[1] for v in acc.values()):5.1f}% of energy)")


def logmel(x, sr=24000, nfft=1024, hop=256, nb=64):
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), nb + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((nb, nfft // 2 + 1), dtype=np.float32)
    for i in range(nb):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        fb[i, lo:mid] = (np.arange(lo, mid) - lo) / max(mid - lo, 1)
        fb[i, mid:hi] = (hi - np.arange(mid, hi)) / max(hi - mid, 1)
    w = np.hanning(nfft).astype(np.float32)
    frames = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * w, axis=1)) ** 2
    return np.log(spec @ fb.T + 1e-8)


m_ref, m_got = logmel(ref), logmel(got)
d = m_got - m_ref
print(f"\nlog-mel (64 bands, 10.7 ms hop): MAE {np.abs(d).mean():.3f}  "
      f"RMSE {np.sqrt((d ** 2).mean()):.3f}  corr {np.corrcoef(m_got.ravel(), m_ref.ravel())[0, 1]:.5f}")
# reference point: white noise at the same SNR (is the error noise-like or structured?)
rng = np.random.default_rng(0)
noise = rng.standard_normal(n).astype(np.float32)
noise *= np.sqrt((ref ** 2).sum() / ((noise ** 2).sum() * 10 ** (snr(got, ref) / 10)))
m_noise = logmel(ref + noise)
dn = m_noise - m_ref
print(f"  white noise @ same SNR: log-mel MAE {np.abs(dn).mean():.3f} (lower = our error is more benign)")
print("  (log-mel MAE < 0.1 ≈ transparent, 0.3-0.5 ≈ clearly coloured, >1 ≈ heavy artefacts)")
