#!/usr/bin/env python3
"""Does the mask survive export_llm_backbone.py's export path?

Builds a tiny randomly-initialised Qwen3 (no download, no checkpoint) and runs
exactly the same route as the real export:
    torch.export.export(wrapper, (embeds, mask4d))
    torch.onnx.export(exported_program, args, path, dynamo=True)
then counts how many nodes actually consume the attention_mask input.

If the mask is dead here as well, the problem is the export path, not qai-hub.
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModel

OUT = "/data/data/com.termux/files/usr/tmp/mask_probe.onnx"
SEQ = 16
HID = 64


class W(torch.nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, inputs_embeds, attention_mask):
        return self.llm(inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask, return_dict=True)[0]


def main():
    cfg = AutoConfig.for_model(
        "qwen3", vocab_size=128, hidden_size=HID, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=HID // 4, max_position_embeddings=512, tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    llm = AutoModel.from_config(cfg)
    mod = W(llm).eval()

    emb = torch.randn(1, SEQ, HID)
    mask = torch.zeros(1, 1, SEQ, SEQ, dtype=torch.float32)
    with torch.no_grad():
        ref = mod(emb, mask)
        # a -1e9 block in the mask must change what the model computes
        m2 = mask.clone()
        m2[0, 0, :, SEQ // 2:] = -1e9
        alt = mod(emb, m2)
    print("torch: delta between two masks %.4f" % (ref - alt).abs().max().item())

    ep = torch.export.export(mod, (emb, mask))
    torch.onnx.export(ep, (emb, mask), OUT, input_names=["inputs_embeds", "attention_mask"],
                      output_names=["hidden_states"], dynamo=True)

    import onnx
    m = onnx.load(OUT)
    users = {}
    for n in m.graph.node:
        for i in n.input:
            users.setdefault(i, []).append(n.op_type)
    print("inputs:", [i.name for i in m.graph.input])
    for name in ("attention_mask", "inputs_embeds"):
        print(f"{name}: {len(users.get(name, []))} consumers -> {users.get(name, [])[:6]}")
    print("nodes:", len(m.graph.node))
    print("size %.1f MB" % (os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
