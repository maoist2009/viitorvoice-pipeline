#!/usr/bin/env python3
"""Local (Termux / CPU) export of w2v_bert_2_0.onnx via the dynamo ONNX path.

The legacy TorchScript-based torch.onnx.export (dynamo=False) is broken on
torch 2.11 (Termux aarch64): 44 prim::Constant nodes carry C++-unreadable
attribute types, so the ONNX symbolic layer cannot read their values.

The working path is:
  torch.export.export(nn.Module, args)  -> ExportedProgram (FX IR)
  torch.onnx.export(exported_program, args, dynamo=True)  -> ONNX

This uses the on-disk torchlib decomposer registry, so no onnxruntime and no
`pip install` are required. The model accepts a raw 16 kHz float waveform and
emits the penultimate transformer block's hidden states (1, T, 1024).

Graph:
  input_values  (1, F)  raw 16 kHz waveform
    -> MelFrontend (log-mel, frame-stack x2 -> 160)   (1, T, 160)
    -> Wav2Vec2BertModel(input_features=..., output_hidden_states=True)
    -> hidden_states[output_idx]  (1, T, 1024)
  C++ side does (feat-mean)/std, transpose(1,2), avg_pool1d(k=2).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mel_frontend import MelFrontend

MODEL_DIR = "viitor-voice-nar/local_models/dualcodec/w2v-bert-2.0"
OUT = "viitor-voice-nar/local_models/dualcodec/w2v-bert-2.0/w2v_bert_2_0.onnx"
OUTPUT_IDX = -2  # penultimate transformer block (upstream default)


class W2VWrapper(torch.nn.Module):
    def __init__(self, mel, m, idx):
        super().__init__()
        self.mel = mel
        self.m = m
        self.idx = idx

    def forward(self, input_values):
        feats = self.mel(input_values)            # (1, T, 160)
        out = self.m(input_features=feats, output_hidden_states=True)
        return out.hidden_states[self.idx]        # (1, T, 1024)


def main() -> int:
    from transformers import Wav2Vec2BertModel
    import onnx
    from onnx import checker

    print(f"[load] {MODEL_DIR}")
    model = Wav2Vec2BertModel.from_pretrained(MODEL_DIR, torchscript=False)
    model.eval()
    mel = MelFrontend().eval()

    wrapper = W2VWrapper(mel, model, OUTPUT_IDX)
    wrapper.eval()

    dummy = (torch.zeros(1, 16000, dtype=torch.float32),)

    print("[export] torch.export.export ...")
    ep = torch.export.export(wrapper, dummy)
    print(f"  ExportedProgram nodes: {len(list(ep.graph.nodes))}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("[onnx] torch.onnx.export(ep, dynamo=True) ...")
    # Let the dynamo path pick the opset it needs (>=17 for the attention /
    # layernorm ops it emits); do not force opset 14 (version_converter assert).
    torch.onnx.export(
        ep, dummy, OUT,
        input_names=["input_values"], output_names=["hidden_states"],
        dynamo=True,
        dynamic_axes={"input_values": {0: "batch", 1: "samples"},
                      "hidden_states": {0: "batch", 1: "frames"}},
    )
    print(f"[onnx] wrote {OUT}")

    # Validate from the file path (in-memory check trips the 2GiB protobuf cap).
    print("[onnx] checker.check_model(from path) ...")
    checker.check_model(OUT)
    print("[onnx] check_model PASS")

    m = onnx.load(OUT)
    print("[onnx] ir_version", m.ir_version,
          "opset", [(o.domain, o.version) for o in m.opset_import])
    print("[onnx] input ", [(i.name, [d.dim_value or d.dim_param
          for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print("[onnx] output", [(o.name, [d.dim_value or d.dim_param
          for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
    return 0


if __name__ == "__main__":
    sys.exit(main())
