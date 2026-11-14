#!/usr/bin/env python3
"""Export the LLM backbone with rope_theta forced to 1e6 and check ORT vs torch.

Builds the Qwen3 config exactly like export_llm_backbone.py but pins
config.rope_theta = 1e6 (transformers 4.57 reads `rope_theta`, NOT
`rope_parameters['rope_theta']`, else it silently falls back to 10000).

Then runs the exported graph in ORT with emb256.npy and compares against the
torch hidden states saved by rope_theta_check.py (hs_theta1m.npz / hs_theta10k.npz).
"""
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel

M = os.path.expanduser("~/vvnar/llm/0p6_emotion")
WORK = os.path.expanduser("~/work")
OUT = os.path.join(WORK, "llm_theta1m.onnx")
SEQ = 256
THETA = float(os.environ.get("THETA", "1000000"))


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    cfg_json = json.load(open(os.path.join(M, "config.json")))
    lc = dict(cfg_json["llm_config"])
    mt = lc.pop("model_type")
    lc.pop("architectures", None)
    cfg = AutoConfig.for_model(mt, **lc)
    cfg.rope_theta = THETA
    cfg.rope_parameters = {"rope_theta": THETA, "rope_type": "default"}
    cfg._attn_implementation = "eager"
    print(f"[cfg] rope_theta={cfg.rope_theta} hidden={cfg.hidden_size}", flush=True)

    llm = AutoModel.from_config(cfg, torch_dtype=torch.float32)
    state = {}
    with safe_open(os.path.join(M, "model.safetensors"), framework="pt") as h:
        for k in h.keys():
            if k.startswith("llm."):
                state[k[4:]] = h.get_tensor(k)
    missing, unexpected = llm.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "rotary_emb.inv_freq"]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    llm.eval()

    class W(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.llm = m

        def forward(self, inputs_embeds, attention_mask):
            return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]

    mod = W(llm).eval()
    dummy = (torch.randn(1, SEQ, cfg.hidden_size), torch.zeros(1, 1, SEQ, SEQ))
    print("[export] ...", flush=True)
    ep = torch.export.export(mod, dummy)
    torch.onnx.export(ep, dummy, OUT, input_names=["inputs_embeds", "attention_mask"],
                      output_names=["hidden_states"], dynamo=True)
    import onnx
    from onnx import numpy_helper
    m = onnx.load(OUT)
    for t in m.graph.initializer:
        arr = numpy_helper.to_array(t, base_dir=os.path.dirname(OUT))
        t.CopyFrom(numpy_helper.from_array(arr, name=t.name))
    onnx.save(m, OUT)
    onnx.checker.check_model(OUT)
    print(f"[ok] {OUT} {os.path.getsize(OUT)/1e6:.0f} MB", flush=True)
    del llm, mod, ep, state, m
    import gc
    gc.collect()

    import onnxruntime as ort
    emb = np.load(os.path.join(WORK, "emb256.npy")).astype(np.float32).reshape(1, SEQ, 1024)
    mask = np.zeros((1, 1, SEQ, SEQ), dtype=np.float32)
    s_ = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    got = s_.run(None, {"inputs_embeds": emb, "attention_mask": mask})[0].reshape(SEQ, 1024)
    got = got.astype(np.float64)

    for tag in ("theta1m", "theta10k"):
        p = os.path.join(WORK, f"hs_{tag}.npz")
        if not os.path.exists(p):
            continue
        ref = np.load(p)["h28"]
        c = (got * ref).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(ref, axis=-1))
        print(f"ORT(export theta={THETA:.0f}) vs torch[{tag}]: cos mean {c.mean():.6f} "
              f"min {c.min():.6f}  max|d| {np.abs(got - ref).max():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
