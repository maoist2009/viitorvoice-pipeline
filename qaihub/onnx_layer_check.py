#!/usr/bin/env python3
"""Which layer of the exported ONNX stops matching torch?

The 28-layer ONNX disagrees with the torch model (cos 0.697) while the single
layer slice matched exactly, so cut the exported graph at the end of every
layer, run each piece with ORT and compare against torch's hidden states.
"""
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
LAYERS = (1, 2, 3, 4, 8, 16, 28)


def torch_hidden():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                      num_hidden_layers=28, num_attention_heads=16,
                      num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
                      rope_theta=1000000, max_position_embeddings=40960,
                      tie_word_embeddings=False, attention_bias=False, hidden_act="silu")
    cfg._attn_implementation = "eager"
    m = Qwen3Model(cfg).to(torch.float32)
    sd = {}
    with safe_open(os.path.join(P.MODEL, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("llm.") and "embed_tokens" not in k:
                sd[k[4:]] = f.get_tensor(k).float()
    m.load_state_dict(sd, strict=False)
    m.eval()
    emb = torch.from_numpy(np.load(os.path.join(W, "emb256.npy")).astype(np.float32))
    with torch.no_grad():
        out = m(inputs_embeds=emb, attention_mask=torch.zeros(1, 1, SEQ, SEQ,
                                                              dtype=torch.bool),
                output_hidden_states=True)
    hs = [h[0].numpy().astype(np.float64) for h in out.hidden_states]
    print(f"torch hidden states {len(hs)}, last layer max|h| {np.abs(hs[-1]).max():.1f}",
          flush=True)
    return hs


def layer_cut_points(onnx_path, n_layers):
    """Residual stream entering layer k+1 == output of layer k.

    Each RMSNorm is Pow->ReduceMean->Add->Sqrt->Reciprocal->Mul(x, rsqrt)
    -> Mul(weight, ·).  So from the `llm.layers.{k}.input_layernorm.weight`
    Mul, input[1] is the Mul(x, rsqrt) node and its input[0] is the residual
    stream.
    """
    import onnx
    m = onnx.load(onnx_path)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cuts = {}
    for li in range(n_layers + 1):
        wname = "llm.norm.weight" if li == n_layers else f"llm.layers.{li}.input_layernorm.weight"
        mulw = next(n for n in g.node
                    if n.op_type == "Mul" and wname in n.input)
        scaled = prod[mulw.input[1]]
        x = scaled.input[0]
        cuts[li] = x
    return cuts


def main():
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    hs = torch_hidden()
    cuts = layer_cut_points(ONNX, 28)
    import onnx.utils
    import onnxruntime as ort
    emb = np.load(os.path.join(W, "emb256.npy")).astype(np.float32).reshape(1, SEQ, 1024)
    mask = np.zeros((1, 1, SEQ, SEQ), dtype=np.float32)
    for kk in (0, 1, 2, 3, 7, 15, 27):
        tmp = os.path.join(W, "cut_%d.onnx" % kk)
        out_name = "hidden_states" if kk == 28 else cuts[kk + 1]
        if kk == 27:
            onnx.utils.extract_model(ONNX, tmp, ["inputs_embeds", "attention_mask"],
                                     ["hidden_states"])
            ref = hs[28]
        else:
            onnx.utils.extract_model(ONNX, tmp, ["inputs_embeds", "attention_mask"],
                                     [out_name])
            ref = hs[kk + 1]
        s_ = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        got = s_.run(None, {"inputs_embeds": emb, "attention_mask": mask})[0]
        got = got.reshape(SEQ, 1024).astype(np.float64)
        c = (got * ref).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(ref, axis=-1))
        print(f"after layer {kk:2d}: cos mean {c.mean():.6f} min {c.min():.6f} "
              f"max|d| {np.abs(got-ref).max():.4f}  max|h| torch {np.abs(ref).max():.0f}",
              flush=True)
        os.remove(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
