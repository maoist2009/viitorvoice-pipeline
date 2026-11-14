#!/usr/bin/env python3
"""Tiny Qwen3-backbone probe: is the qai-hub fp16 DLC faithful at all?

Same op pattern as the real 28-layer export (GQA + RoPE + RMSNorm + SwiGLU,
additive 4-D attention mask), but small enough to upload in seconds:
hidden 256, 2 layers, 4 heads / 2 kv heads, head_dim 64, seq 64.

usage: probe_tiny.py export|submit|poll|link|poll-link|fetch|check
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
W = "/data/data/com.termux/files/usr/tmp/probe_tiny"
PT2 = os.path.join(W, "tiny.pt2")
ONNX = os.path.join(W, "tiny.onnx")
STATE = os.path.join(HERE, "probe_tiny.json")
OPTS = ("--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type float16 "
        "--qairt_version 2.51")
SEQ, HID = 64, 256
EMB = os.path.join(W, "emb.raw")
ATTN = os.path.join(W, "attn.raw")


def build():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=HID, intermediate_size=768,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=64, rms_norm_eps=1e-6,
                      rope_theta=1000000, max_position_embeddings=40960,
                      tie_word_embeddings=False, attention_bias=False,
                      hidden_act="silu")
    cfg._attn_implementation = "eager"
    m = Qwen3Model(cfg).to(torch.float32).eval()
    # the wrapper only ever feeds inputs_embeds; drop the 155 MB embedding table
    m.embed_tokens = torch.nn.Identity()
    return m


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, inputs_embeds, attention_mask):
        return self.m(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                      return_dict=True)[0]


def main():
    cmd = sys.argv[1]
    os.makedirs(W, exist_ok=True)
    if cmd == "export":
        torch.set_grad_enabled(True)
        m = build()
        mod = Wrapper(m).eval()
        emb = torch.randn(1, SEQ, HID)
        mask = torch.zeros(1, 1, SEQ, SEQ)
        ep = torch.export.export(mod, (emb, mask))
        torch.export.save(ep, PT2)
        print("pt2", os.path.getsize(PT2) / 1e6, "MB")
        torch.onnx.export(ep, (emb, mask), ONNX, input_names=["inputs_embeds",
                          "attention_mask"], output_names=["hidden_states"], dynamo=True)
        print("onnx", os.path.getsize(ONNX) / 1e6, "MB")
        emb.numpy().astype(np.float32).tofile(EMB)
        mask.numpy().astype(np.float32).tofile(ATTN)
        import onnxruntime as ort
        s = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
        out = s.run(None, {"inputs_embeds": emb.numpy(), "attention_mask": mask.numpy()})[0]
        np.save(os.path.join(W, "ref_ort.npy"), out)
        # torch reference too (must equal ORT bit-for-bit-ish)
        with torch.no_grad():
            ref = mod(emb, mask).numpy()
        print("ort-vs-torch max|d| %.2e" % np.abs(out - ref).max())
        np.save(os.path.join(W, "ref_torch.npy"), ref)
    elif cmd == "submit":
        import qai_hub as hub
        st = json.load(open(STATE)) if os.path.exists(STATE) else {}
        if "model" not in st:
            m = hub.upload_model(PT2)
            st["model"] = m.model_id
            json.dump(st, open(STATE, "w"), indent=1)
            print("MODEL", st["model"], flush=True)
        if "compile" not in st:
            j = hub.submit_compile_job(model=hub.get_model(st["model"]),
                                       device=hub.Device("Samsung Galaxy S25 (Family)"),
                                       name="tiny_llm_fp16", options=OPTS)
            st["compile"] = j.job_id
            json.dump(st, open(STATE, "w"), indent=1)
            print("COMPILE", st["compile"], flush=True)
    elif cmd == "poll":
        import qai_hub as hub
        st = json.load(open(STATE))
        print(hub.get_job(st["compile"]).get_status().code)
    elif cmd == "link":
        import qai_hub as hub
        st = json.load(open(STATE))
        if "link" in st:
            print("already", st["link"])
            return
        j = hub.get_job(st["compile"])
        if j.get_status().code != "SUCCESS":
            print("not ready", j.get_status().code)
            return
        lj = hub.submit_link_job(models=j.get_target_model(), device=j.device,
                                 name="tiny_llm_fp16_link",
                                 options="--qairt_version 2.51")
        st["link"] = lj.job_id
        json.dump(st, open(STATE, "w"), indent=1)
        print("LINK", st["link"])
    elif cmd == "poll-link":
        import qai_hub as hub
        st = json.load(open(STATE))
        print(hub.get_job(st["link"]).get_status().code)
    elif cmd == "fetch":
        import qai_hub as hub
        st = json.load(open(STATE))
        out = os.path.join(W, "bin")
        os.makedirs(out, exist_ok=True)
        hub.get_job(st["link"]).get_target_model().download(out)
        print("downloaded", os.listdir(out))
    elif cmd == "check":
        import nar_proto as P
        st = json.load(open(STATE))
        binp = os.path.join(W, "bin", sorted(os.listdir(os.path.join(W, "bin")))[0])
        emb = np.fromfile(EMB, dtype=np.float32).reshape(1, SEQ, HID)
        attn = np.fromfile(ATTN, dtype=np.float32).reshape(1, 1, SEQ, SEQ)
        (hid,), ms = P.net_run(binp, [EMB, ATTN], os.path.join(W, "run"),
                               [("output_0.raw", SEQ * HID * 4)])
        ref = np.load(os.path.join(W, "ref_ort.npy")).reshape(-1)
        got = hid.reshape(-1).astype(np.float64)
        c = float(got @ ref / np.linalg.norm(got) / np.linalg.norm(ref))
        print(f"[check] {ms:.0f} ms cos {c:.6f} "
              f"norm {np.linalg.norm(got):.3f}/{np.linalg.norm(ref):.3f} "
              f"max|d| {np.abs(got-ref).max():.4f}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
