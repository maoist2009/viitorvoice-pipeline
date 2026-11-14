#!/usr/bin/env python3
"""Re-export the ViiTorVoice LLM backbone (Qwen3-0.6B) to fp32 ONNX.

The 2026-07-17 export produced a TRUNCATED file (DecodeError: Truncated
message) because it used the legacy TorchScript torch.onnx.export path,
which is 100% broken on torch 2.11 Termux, plus the export was started
before model.safetensors finished downloading.

Contract mirrors viitorvoice/llm/export_onnx.py:
  inputs_embeds  (N, S, 1024) f32
  attention_mask (N, 1, S, S) f32
  -> hidden_states (N, S, 1024) f32   (last_hidden_state, outputs[0])

Weights are the llm.* prefix of model.safetensors, loaded into
AutoModel.from_config(Qwen3 config) with the prefix stripped.

Static shape: the NPU graph is compiled for a fixed seq len. 256 frames
matches the default export_seq_len of the upstream script.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel

MODEL = os.path.expanduser(
    "~/projects/voice_editor/sm8750_voice_editor/viitor-voice-nar"
    "/local_models/llm/0p6_emotion")
OUT = os.path.expanduser("~/tmp/dl_vv/llm_backbone.onnx")
SEQ_LEN = 256
BATCH = 1


class LLMBackboneWrapper(torch.nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs[0]


def main():
    cfg_path = os.path.join(MODEL, "config.json")
    weights = os.path.join(MODEL, "model.safetensors")
    cfg = json.loads(open(cfg_path).read())
    llm_cfg = dict(cfg["llm_config"])
    model_type = llm_cfg.pop("model_type")
    llm_cfg.pop("architectures", None)
    print(f"[cfg] model_type={model_type} hidden={llm_cfg['hidden_size']}")

    llm_config = AutoConfig.for_model(model_type, **llm_cfg)
    llm_config._attn_implementation = "eager"
    llm = AutoModel.from_config(llm_config, torch_dtype=torch.float32)

    state = {}
    with safe_open(weights, framework="pt", device="cpu") as h:
        for k in h.keys():
            if k.startswith("llm."):
                state[k[len("llm."):]] = h.get_tensor(k)
    print(f"[weights] loaded {len(state)} tensors from llm.* prefix")
    missing, unexpected = llm.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "rotary_emb.inv_freq"]
    if missing or unexpected:
        print(f"[ERROR] missing={missing[:10]} unexpected={unexpected[:10]}")
        sys.exit(1)
    llm.eval()

    mod = LLMBackboneWrapper(llm).eval()
    dummy = (
        torch.randn(BATCH, SEQ_LEN, llm_config.hidden_size, dtype=torch.float32),
        torch.zeros(BATCH, 1, SEQ_LEN, SEQ_LEN, dtype=torch.float32),
    )

    print("[export] torch.export.export ...", flush=True)
    ep = torch.export.export(mod, dummy)
    print(f"  ExportedProgram nodes: {len(list(ep.graph.nodes))}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.onnx.export(
        ep, dummy, OUT,
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["hidden_states"],
        dynamo=True,
    )

    # qai-hub rejects external-data refs, so inline all initializers.
    import onnx
    from onnx import checker, numpy_helper
    m = onnx.load(OUT)
    base_dir = os.path.dirname(OUT)
    for t in m.graph.initializer:
        arr = numpy_helper.to_array(t, base_dir=base_dir)
        t.CopyFrom(numpy_helper.from_array(arr, name=t.name))
    onnx.save(m, OUT)
    checker.check_model(OUT)
    inlined = sum(len(t.raw_data) for t in m.graph.initializer)
    print(f"[ok] {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB, "
          f"{inlined/1e6:.1f} MB inlined)")

    mm = onnx.load(OUT)
    for i in mm.graph.input:
        print("  IN ", i.name, [d.dim_value or d.dim_param
              for d in i.type.tensor_type.shape.dim])
    for o in mm.graph.output:
        print("  OUT", o.name, [d.dim_value or d.dim_param
              for d in o.type.tensor_type.shape.dim])


if __name__ == "__main__":
    main()
