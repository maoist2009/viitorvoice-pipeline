#!/usr/bin/env python3
"""Run the *reference* generator code (viitorvoice/llm/generation.py) with a
torch backbone on the phone, then decode with the official dualcodec decoder.

This is the ground truth for the NAR port: their prompt building, their
schedule, their scoring, their pause handling — only `forward_step` is
re-implemented here with the HF Qwen3 model + the two heads.

usage: torch_generate.py <text> <out.wav> [dtype=fp32|bf16]
"""
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

IMPL = "/data/data/com.termux/files/home/projects/voice_editor/viitor-voice-nar"
sys.path.insert(0, IMPL)
MODEL = ("/data/data/com.termux/files/home/projects/voice_editor/"
         "sm8750_voice_editor/viitor-voice-nar/local_models/llm/0p6_emotion")
CKPT = os.path.join(MODEL, "model.safetensors")
OUTDIR = "/data/data/com.termux/files/usr/tmp/vv_nar"

from viitorvoice.llm.generation import LLMGenerationConfig, ViiTorVoiceLLMGenerator  # noqa: E402
from viitorvoice.llm.text_utils import RuleDurationEstimator  # noqa: E402


class TorchBackbone:
    """Everything ViiTorVoiceLLMGenerator expects from `self.model`."""

    def __init__(self, dtype):
        cfg_json = json.load(open(os.path.join(MODEL, "config.json")))
        self.config = cfg_json
        llm_cfg = cfg_json["llm_config"]
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

        cfg = Qwen3Config(
            vocab_size=llm_cfg["vocab_size"], hidden_size=llm_cfg["hidden_size"],
            intermediate_size=llm_cfg["intermediate_size"],
            num_hidden_layers=llm_cfg["num_hidden_layers"],
            num_attention_heads=llm_cfg["num_attention_heads"],
            num_key_value_heads=llm_cfg["num_key_value_heads"],
            head_dim=llm_cfg["head_dim"], rms_norm_eps=llm_cfg["rms_norm_eps"],
            rope_theta=1_000_000, attention_bias=False, tie_word_embeddings=False)
        self.dtype = dtype
        self.model = Qwen3Model(cfg).to(dtype).eval()
        sd = {}
        with safe_open(CKPT, framework="pt") as f:
            for k in f.keys():
                if k.startswith("llm."):
                    sd[k[4:]] = f.get_tensor(k).to(dtype)
            self.text_emb = f.get_tensor("llm.embed_tokens.weight").to(dtype)
            self.sem_emb = f.get_tensor("semantic_embedding.weight").to(dtype)
            self.aco_emb = f.get_tensor("acoustic_embedding.weight").to(dtype)
            self.sem_head = f.get_tensor("semantic_head.weight").to(dtype)
            self.aco_head = f.get_tensor("acoustic_head.weight").to(dtype)
            self.offsets = f.get_tensor("acoustic_codebook_offsets").to(torch.long)
        self.model.load_state_dict(sd, strict=False)
        self.device = torch.device("cpu")
        self.active_providers = ["CPU"]
        self.num_acoustic_codebooks = int(cfg_json["num_audio_codebook"]) - 1
        self.acoustic_codebook_size = int(cfg_json["audio_codebook_sizes"][1])

    def prepare_embed_inputs(self, input_ids, audio_mask):
        text_embeds = F.embedding(input_ids[:, 0, :], self.text_emb)
        am = audio_mask.unsqueeze(-1)
        sem_ids = torch.where(audio_mask, input_ids[:, 0, :], torch.zeros_like(input_ids[:, 0, :]))
        sem = F.embedding(sem_ids, self.sem_emb)
        aco_ids = torch.where(audio_mask.unsqueeze(1), input_ids[:, 1:, :],
                              torch.zeros_like(input_ids[:, 1:, :]))
        aco = F.embedding(aco_ids + self.offsets.view(1, -1, 1), self.aco_emb).sum(1)
        audio = (sem + aco) * am
        return torch.where(am, audio, text_embeds)

    FILL = os.environ.get("VV_FILL") == "1"

    def _silence(self):
        if not hasattr(self, "_sil"):
            import torch as _t
            d = _t.load("/data/data/com.termux/files/home/projects/voice_editor/"
                        "sm8750_voice_editor/viitor-voice-nar/local_models/assets/"
                        "dualcodec_silence_2s.pt", map_location="cpu")
            t = _t.as_tensor(d["tokens"] if isinstance(d, dict) else d).long()
            self._sil = t
        return self._sil

    @torch.no_grad()
    def forward_step(self, input_ids, audio_mask, attention_mask):
        self._orig_len = None
        if self.FILL:
            b, cb, s_q = input_ids.shape
            self._orig_len = s_q
            need = 256 - s_q
            if need > 0:
                sil = self._silence()
                reps = (need + sil.shape[1] - 1) // sil.shape[1]
                fill = sil.repeat(1, reps)[:, :need].unsqueeze(0)
                input_ids = torch.cat([input_ids, fill], dim=2)
                audio_mask = torch.cat([audio_mask,
                                        torch.ones(1, need, dtype=torch.bool)], dim=1)
                pad = torch.zeros(1, 1, 256, 256, dtype=torch.bool)
                pad[0, 0, :s_q, :s_q] = True
                pad[0, 0, s_q:, s_q:] = True
                pad[0, 0, :s_q, s_q:] = True
                pad[0, 0, s_q:, :s_q] = True
                attention_mask = pad
        emb = self.prepare_embed_inputs(input_ids, audio_mask)
        h = self.model(inputs_embeds=emb, attention_mask=attention_mask,
                       use_cache=False).last_hidden_state
        sem = h @ self.sem_head.T
        aco = (h @ self.aco_head.T).view(h.shape[0], h.shape[1],
                                         self.num_acoustic_codebooks,
                                         self.acoustic_codebook_size)
        if self.FILL:
            s_q = sem.shape[1] - (256 - sem.shape[1]) if False else None
        out_sem, out_aco = sem, aco.permute(0, 2, 1, 3)
        if self.FILL and getattr(self, "_orig_len", None):
            n = self._orig_len
            out_sem = out_sem[:, :n]
            out_aco = out_aco[:, :, :n]
        return types.SimpleNamespace(semantic_logits=out_sem, acoustic_logits=out_aco)


def build_generator(dtype):
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    g = ViiTorVoiceLLMGenerator.__new__(ViiTorVoiceLLMGenerator)
    from pathlib import Path
    g.checkpoint_dir = Path(MODEL)
    g.config = cfg
    g.model = TorchBackbone(dtype)
    g.device = g.model.device
    g.tokenizer = g._load_tokenizer(g.checkpoint_dir)
    g.duration_estimator = RuleDurationEstimator()
    g.frame_rate = int(cfg.get("audio_frame_rate", 25))
    g.num_audio_codebook = int(cfg["num_audio_codebook"])
    g.audio_mask_ids = list(cfg["audio_mask_ids"])
    g.audio_pause_ids = cfg.get("audio_pause_ids")
    g.audio_separator_ids = cfg.get("audio_separator_ids")
    g.audio_valid_vocab_sizes = list(cfg["audio_valid_vocab_sizes"])
    g.active_providers = ["CPU"]
    g._no_ref_text_silence_codec_cache = None
    return g


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "你好，世界。今天我们测试一下这个语音合成系统。"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUTDIR, "ref_out.wav")
    dtype = torch.float32 if (len(sys.argv) < 4 or sys.argv[3] == "fp32") else torch.bfloat16
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)

    g = build_generator(dtype)
    est = g._estimate_target_tokens(text, None, None)
    print(f"[ref] estimated {est} frames ({est/25:.2f}s)", flush=True)
    gen_config = LLMGenerationConfig()
    import time
    t0 = time.perf_counter()
    tokens = g.generate(text, language="zh", generation_config=gen_config)[0]
    print(f"[ref] generated {tuple(tokens.shape)} in {time.perf_counter()-t0:.0f}s", flush=True)
    tk = tokens.numpy().astype(np.int64)
    np.save(os.path.join(OUTDIR, "ref_tokens.npy"), tk)
    print("[ref] sem uniques", len(np.unique(tk[0])), "first 20", tk[0][:20].tolist(), flush=True)

    # decode with the official decoder, packed with silence context
    import onnxruntime as ort
    sil = torch.load(os.path.join(IMPL, "../sm8750_voice_editor/viitor-voice-nar/"
                                  "local_models/assets/dualcodec_silence_2s.pt"),
                     map_location="cpu")
    sil = torch.as_tensor(sil["tokens"] if isinstance(sil, dict) else sil).long().numpy()
    gap = sil[:, :25]
    packed = np.concatenate([gap, tk, gap], axis=1)[None]
    sess = ort.InferenceSession(
        "/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/"
        "viitor-voice-nar/local_models/dualcodec/dualcodec_ckpts/dualcodec_decoder.onnx",
        providers=["CPUExecutionProvider"])
    audio = sess.run(None, {"semantic_codes": packed[:, :1].astype(np.int64),
                            "acoustic_codes": packed[:, 1:].astype(np.int64)})[0].reshape(-1)
    seg = audio[25 * 960: (25 + tk.shape[1]) * 960 - 4]
    import wave
    pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16)
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(pcm.tobytes())
    n = 2400
    env = [float(np.sqrt((seg[i:i+n]**2).mean())) for i in range(0, len(seg)-n, n)]
    print("[ref] wav %.2fs rms %.4f" % (len(seg)/24000, np.sqrt((seg**2).mean())), flush=True)
    print("[ref] env", " ".join("%.2f" % x for x in env[:40]), flush=True)


if __name__ == "__main__":
    main()
