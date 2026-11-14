"""Faithful torch re-implementation of SeamlessM4TFeatureExtractor (w2v-bert-2.0 front).

Ported exactly from transformers.audio_utils (SeamlessM4TFeatureExtractor /
mel_filter_bank / spectrogram) so the exported ONNX can take a raw 16 kHz
waveform and keep the C++ W2VFrontend::extract(raw_16k_waveform) contract.

Reference params (SeamlessM4TFeatureExtractor + _extract_fbank_features):
  - waveform *= 2**15  (Kaldi 16-bit scaling)
  - frame_length=400, hop_length=160, fft_length=512, onesided -> 257 bins
  - center=False, remove_dc_offset=True, preemphasis=0.97
  - window = povey(400, periodic=False) zero-padded to frame_length
  - mel_filter_bank(num_frequency_bins=257, num_mel_filters=80, min_freq=20,
                    max_freq=8000, norm=None, mel_scale="kaldi",
                    triangularize_in_mel_space=True)
  - mel_floor=1.192092955078125e-07, log_mel="log" (natural log)
  - per-mel zero-mean / unit-var norm over valid frames (np.var ddof=0)
  - frame stacking stride=2 -> output feature dim 160
"""
import math
import torch
import torch.nn as nn


def _povey_window(n: int) -> torch.Tensor:
    # window_function(400, "povey", periodic=False) -> hann(length)**0.85, length = n (not periodic)
    length = n
    w = torch.hann_window(length, periodic=False)
    return torch.pow(w, 0.85)


def _kaldi_mel_filterbank(num_freq_bins: int, num_mel_filters: int,
                          min_freq: float, max_freq: float, sr: int,
                          device) -> torch.Tensor:
    # mel_filter_bank with mel_scale="kaldi", triangularize_in_mel_space=True, norm=None
    def hz_to_mel(hz):
        return 1127.0 * torch.log(1.0 + hz / 700.0)

    mel_min = hz_to_mel(torch.tensor(min_freq))
    mel_max = hz_to_mel(torch.tensor(max_freq))
    mel_freqs = torch.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs_hz = 700.0 * (torch.exp(mel_freqs / 1127.0) - 1.0)

    fft_bin_width = sr / ((num_freq_bins - 1) * 2)
    fft_freqs = hz_to_mel(fft_bin_width * torch.arange(num_freq_bins))
    filter_freqs = mel_freqs  # triangularize in mel space

    filter_diff = filter_freqs[1:] - filter_freqs[:-1]
    slopes = filter_freqs.unsqueeze(0) - fft_freqs.unsqueeze(1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1].unsqueeze(0)
    up_slopes = slopes[:, 2:] / filter_diff[1:].unsqueeze(0)
    fb = torch.maximum(torch.zeros(1), torch.minimum(down_slopes, up_slopes))
    return fb.to(device)  # (num_freq_bins, num_mel_filters)


class MelFrontend(nn.Module):
    def __init__(self, sr: int = 16000, frame_length: int = 400, hop: int = 160,
                 fft_length: int = 512, n_mels: int = 80, preemph: float = 0.97,
                 mel_floor: float = 1.192092955078125e-07, remove_dc_offset: bool = True):
        super().__init__()
        self.sr = sr
        self.frame_length = frame_length
        self.hop = hop
        self.fft_length = fft_length
        self.n_mels = n_mels
        self.preemph = preemph
        self.mel_floor = mel_floor
        self.remove_dc_offset = remove_dc_offset

        window = _povey_window(frame_length)
        self.register_buffer("window", window, persistent=False)
        fb = _kaldi_mel_filterbank(fft_length // 2 + 1, n_mels, 20.0,
                                   sr // 2, sr, torch.device("cpu"))
        self.register_buffer("fb", fb, persistent=False)  # (n_mels, num_freq_bins)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (1, N) raw float (kept float32 for ONNX traceability)
        x = waveform.squeeze(0).to(torch.float32) * (2 ** 15)  # Kaldi 16-bit scaling

        # Framed FFT via unfold (fully dynamic / traceable, no Python loop).
        # Reference steps frames by frame_length (400) with fft buffer 512.
        frames = x.unfold(0, self.frame_length, self.hop)  # (T, frame_length)
        if self.remove_dc_offset:
            frames = frames - frames.mean(dim=1, keepdim=True)
        if self.preemph is not None:
            first = frames[:, :1] * (1.0 - self.preemph)
            rest = frames[:, 1:] - self.preemph * frames[:, :-1]
            frames = torch.cat([first, rest], dim=1)
        frames = frames * self.window.unsqueeze(0)              # (T, frame_length)
        frames = torch.nn.functional.pad(
            frames, (0, self.fft_length - self.frame_length))   # (T, fft_length)
        spec = torch.fft.rfft(frames, dim=1)                    # (T, num_freq_bins)
        power = spec.abs().pow(2).transpose(0, 1)               # (num_freq_bins, T)
        # mel: (n_mels, num_freq_bins) @ (num_freq_bins, T) -> (n_mels, T)
        mel = torch.matmul(self.fb.to(power.dtype).transpose(0, 1), power)
        mel = torch.clamp(mel, min=self.mel_floor)
        log_mel = torch.log(mel)  # (n_mels, T)
        # per-mel zero-mean / unit-var over time (ddof=0)
        mean = log_mel.mean(dim=1, keepdim=True)
        std = log_mel.std(dim=1, unbiased=False, keepdim=True)
        log_mel = (log_mel - mean) / (std + 1e-7)  # (n_mels, T)
        # frame stacking stride 2 -> (T', 160)
        log_mel = log_mel.transpose(0, 1)  # (T, n_mels)
        T = log_mel.size(0)
        T2 = (T - 1) // 2
        stacked = torch.cat([log_mel[:T2 * 2:2], log_mel[1:T2 * 2:2]], dim=1)
        return stacked.unsqueeze(0).to(torch.float32)  # (1, T2, 160)
