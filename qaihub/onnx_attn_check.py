#!/usr/bin/env python3
"""Compare layer-0 attention internals (q/k after rope, probs) with torch."""
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


def build():
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
    return m


def cos_stats(a, b, label):
    a = np.asarray(a).reshape(-1, a.shape[-1]).astype(np.float64)
    b = np.asarray(b).reshape(-1, b.shape[-1]).astype(np.float64)
    if a.shape != b.shape:
        print(f"{label}: shape {a.shape} vs {b.shape}")
        return
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    print(f"{label}: cos mean {c.mean():.6f} min {c.min():.6f} max|d| "
          f"{np.abs(a-b).max():.4f}  (absmax {np.abs(b).max():.1f})")


def main():
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    m = build()
    L0 = m.layers[0]
    A = L0.self_attn
    emb = torch.from_numpy(np.load(os.path.join(W, "emb256.npy")).astype(np.float32))
    caps = {}
    L0.input_layernorm.register_forward_hook(
        lambda mod, i, o: caps.__setitem__("norm", o.detach()))
    with torch.no_grad():
        m(inputs_embeds=emb, attention_mask=torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bool))
    x = caps["norm"]                        # (1, 256, 1024)
    q = A.q_proj(x).view(1, SEQ, 16, 128)
    k = A.k_proj(x).view(1, SEQ, 8, 128)
    v = A.v_proj(x).view(1, SEQ, 8, 128)
    qn = A.q_norm(q).transpose(1, 2)        # (1, 16, 256, 128)
    kn = A.k_norm(k).transpose(1, 2)        # (1, 8, 256, 128)
    pos = torch.arange(SEQ)[None]
    cos, sin = m.rotary_emb(x, pos)
    with torch.no_grad():
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        qr, kr = apply_rotary_pos_emb(qn, kn, cos, sin)
    vs = v.transpose(1, 2)
    keys = torch.repeat_interleave(kn, 2, dim=1)
    vals = torch.repeat_interleave(vs, 2, dim=1)
    # pre-rope scores
    import onnx.utils
    import onnxruntime as ort
    embn = emb.numpy().reshape(1, SEQ, 1024)
    mask = np.zeros((1, 1, SEQ, SEQ), dtype=np.float32)
    for name, ref, label in (("mul_5", qn.transpose(1, 2).reshape(1, SEQ, 2048), "q after q_norm"),
                             ("add_3", qr, "q after rope"),
                             ("add_4", kr, "k after rope"),
                             ("mul_12", None, "scores (pre-mask)"),
                             ("val_117", None, "probs")):
        tmp = os.path.join(W, "cut_x.onnx")
        onnx.utils.extract_model(ONNX, tmp, ["inputs_embeds", "attention_mask"], [name])
        s = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        out = s.run(None, {"inputs_embeds": embn, "attention_mask": mask})[0]
        os.remove(tmp)
        if ref is None:
            if name == "mul_12":
                sc = (qr @ kr.transpose(-1, -2)) * (128 ** -0.5)
                cos_stats(out[0][:, :8], sc[0][:, :8], "scores vs torch(all-fp32 recompute)")
            else:
                pr = torch.softmax((qr @ kr.transpose(-1, -2)) * (128 ** -0.5), -1)
                cos_stats(out.reshape(-1, SEQ), pr[0].reshape(-1, SEQ), "probs vs torch")
                print("   probs absmax %.4f  torch absmax %.4f"
                      % (np.abs(out).max(), float(pr.abs().max())))
            continue
        cos_stats(out, np.asarray(ref[0] if hasattr(ref, "shape") and ref.dim() == 4
                                  else ref), label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
