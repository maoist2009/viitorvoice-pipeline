#!/usr/bin/env python3
"""Export truncated Qwen3 backbones (first N layers) as ONNX.

Used to bisect where the compiled graph stops matching torch: 1 layer, 4
layers, then the full 28.  Same wrapper as export_llm_backbone.py so the
contract (inputs_embeds, attention_mask -> last_hidden_state) is identical.

usage: layer_probe.py <n_layers> [out.onnx]
"""
import json
import os
import sys

import torch
from safetensors import safe_open

MODEL = os.path.expanduser(
    "~/projects/voice_editor/sm8750_voice_editor/viitor-voice-nar"
    "/local_models/llm/0p6_emotion")
SEQ = 256


class W(torch.nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, inputs_embeds, attention_mask):
        return self.llm(inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask, return_dict=True)[0]


def build(n_layers):
    cfg_all = json.load(open(os.path.join(MODEL, "config.json")))["llm_config"]
    cfg_all.pop("architectures", None)
    mt = cfg_all.pop("model_type")
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.for_model(mt, **cfg_all)
    cfg.num_hidden_layers = n_layers
    cfg._attn_implementation = "eager"
    llm = AutoModel.from_config(cfg, torch_dtype=torch.float32)
    sd = {}
    with safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if not k.startswith("llm."):
                continue
            name = k[4:]
            if name.startswith("layers.") and int(name.split(".")[1]) >= n_layers:
                continue
            sd[name] = f.get_tensor(k).float()
    missing, unexpected = llm.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary_emb.inv_freq" not in m]
    print(f"[n={n_layers}] missing {missing[:5]} unexpected {list(unexpected)[:5]}", flush=True)
    return llm.eval(), cfg


def main():
    n = int(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else f"/data/data/com.termux/files/usr/tmp/llm_{n}l.onnx"
    llm, cfg = build(n)
    mod = W(llm).eval()
    dummy = (torch.randn(1, SEQ, cfg.hidden_size),
             torch.zeros(1, 1, SEQ, SEQ, dtype=torch.float32))
    ep = torch.export.export(mod, dummy)
    torch.onnx.export(ep, dummy, out, input_names=["inputs_embeds", "attention_mask"],
                      output_names=["hidden_states"], dynamo=True)
    import onnx
    from onnx import numpy_helper
    m = onnx.load(out)
    base = os.path.dirname(out)
    for t in m.graph.initializer:
        arr = numpy_helper.to_array(t, base_dir=base)
        t.CopyFrom(numpy_helper.from_array(arr, name=t.name))
    onnx.save(m, out)
    onnx.checker.check_model(out)
    print(f"[ok] {out} {os.path.getsize(out)/1e6:.1f} MB", flush=True)

    # torch reference hidden for the same weights, saved for the device compare
    emb = torch.from_numpy(__import__("numpy").load(
        "/data/data/com.termux/files/usr/tmp/vv_nar/emb256.npy").astype("float32").reshape(1, SEQ, -1))
    with torch.no_grad():
        h = mod(emb, torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bool))
    __import__("numpy").save(f"/data/data/com.termux/files/usr/tmp/vv_nar/torch_hid_{n}l.npy",
                             h.numpy().reshape(SEQ, -1).astype("float32"))
    print("[ok] torch reference saved", flush=True)


if __name__ == "__main__":
    main()
