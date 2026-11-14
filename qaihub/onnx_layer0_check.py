#!/usr/bin/env python3
"""Bisect layer 0 of the exported ONNX against torch submodule outputs."""
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402

W = "/data/data/com.termux/files/usr/tmp/vv_nar"
ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_backbone.onnx")
SEQ = 256


def torch_refs():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                      num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                      head_dim=128, rms_norm_eps=1e-6, rope_theta=1000000,
                      max_position_embeddings=40960, tie_word_embeddings=False,
                      attention_bias=False, hidden_act="silu")
    cfg._attn_implementation = "eager"
    m = Qwen3Model(cfg).to(torch.float32)
    sd = {}
    with safe_open(os.path.join(P.MODEL, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("llm.") and "embed_tokens" not in k:
                sd[k[4:]] = f.get_tensor(k).float()
    m.load_state_dict(sd, strict=False)
    m.eval()
    L0 = m.layers[0]
    got = {}

    def h_norm(mod, i, o):
        got["norm"] = o.detach()[0]

    def h_attn(mod, i, o):
        got["attn"] = o[0].detach()[0]

    def h_mlp(mod, i, o):
        got["mlp"] = o.detach()[0]

    L0.input_layernorm.register_forward_hook(h_norm)
    L0.self_attn.register_forward_hook(h_attn)
    L0.mlp.register_forward_hook(h_mlp)
    emb = torch.from_numpy(np.load(os.path.join(W, "emb256.npy")).astype(np.float32))
    with torch.no_grad():
        m(inputs_embeds=emb, attention_mask=torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bool))
    return {k: v.numpy().astype(np.float64) for k, v in got.items()}


def main():
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    refs = torch_refs()
    import onnx.utils
    import onnxruntime as ort
    emb = np.load(os.path.join(W, "emb256.npy")).astype(np.float32).reshape(1, SEQ, 1024)
    mask = np.zeros((1, 1, SEQ, SEQ), dtype=np.float32)
    cuts = [("mul_3", refs["norm"], "input_layernorm out"),
            ("add_6", refs["norm"] + refs["attn"], "residual after attn"),
            ("add_8", refs["norm"] + refs["attn"] + refs["mlp"], "residual after mlp"),
            ("add_3", None, "q after rope"),
            ("add_4", None, "k after rope"),
            ("val_117", None, "attention probs"),
            ("matmul_2", None, "attn*v"),
            ("linear_3", refs["attn"], "o_proj")]
    for name, ref, label in cuts:
        tmp = os.path.join(W, "cut_l0.onnx")
        try:
            onnx.utils.extract_model(ONNX, tmp, ["inputs_embeds", "attention_mask"], [name])
        except Exception as e:
            print(f"{label:22s} extract failed: {e}")
            continue
        s = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        out = s.run(None, {"inputs_embeds": emb, "attention_mask": mask})[0]
        os.remove(tmp)
        if ref is None:
            print(f"{label:22s} {name}: shape {out.shape} "
                  f"absmax {np.abs(out).max():.3f} mean {out.mean():.4f}")
            continue
        a = np.asarray(out).reshape(-1, out.shape[-1]).astype(np.float64)
        b = np.asarray(ref).reshape(-1, ref.shape[-1]).astype(np.float64)
        if a.shape != b.shape:
            print(f"{label:22s} {name}: shape {a.shape} vs torch {b.shape}")
            continue
        c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        print(f"{label:22s} {name}: cos mean {c.mean():.6f} min {c.min():.6f} "
              f"max|d| {np.abs(a - b).max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
