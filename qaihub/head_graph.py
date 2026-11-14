#!/usr/bin/env python3
"""Build the ViiTorVoice output-head graph and push it to qai-hub.

The compiled LLM graph stops at hidden_states [1,256,1024]; the codebooks are
produced by two bias-free Linear layers that were never exported:

    semantic_head  (16387, 1024)   -> semantic logits
    acoustic_head  (11297, 1024)   -> 11 x 1027 acoustic logits

Both are plain matmuls, so the ONNX is built directly with onnx.helper (no
torch needed) and kept in fp16: QNN has no fp32 compute path, and one matmul in
fp16 is numerically transparent. IO stays fp32 (Cast at the boundary) so the
app can feed the LLM's fp32 output straight in.

usage: head_graph.py export | submit | poll | link | poll-link | fetch
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.expanduser(
    "~/projects/voice_editor/sm8750_voice_editor/viitor-voice-nar/"
    "local_models/llm/0p6_emotion/model.safetensors")
ONNX_PATH = os.path.join(HERE, "head_fp16.onnx")
STATE = os.path.join(HERE, "head_job.json")
SEQ, HID = 256, 1024


def export():
    with safe_open(CKPT, framework="np") as f:
        sem = f.get_tensor("semantic_head.weight")     # (16387, 1024) f32
        aco = f.get_tensor("acoustic_head.weight")     # (11297, 1024) f32
    print("sem", sem.shape, "aco", aco.shape, flush=True)

    # Linear == x @ W.T ; store W.T as fp16 to halve the file.
    inits = [
        numpy_helper.from_array(np.ascontiguousarray(sem.T).astype(np.float16), "sem_w"),
        numpy_helper.from_array(np.ascontiguousarray(aco.T).astype(np.float16), "aco_w"),
    ]
    nodes = [
        helper.make_node("Cast", ["hidden"], ["h16"], to=TensorProto.FLOAT16),
        helper.make_node("MatMul", ["h16", "sem_w"], ["sem16"]),
        helper.make_node("MatMul", ["h16", "aco_w"], ["aco16"]),
        helper.make_node("Cast", ["sem16"], ["semantic_logits"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["aco16"], ["acoustic_logits"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes, "vv_head",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, SEQ, HID])],
        [helper.make_tensor_value_info("semantic_logits", TensorProto.FLOAT,
                                       [1, SEQ, sem.shape[0]]),
         helper.make_tensor_value_info("acoustic_logits", TensorProto.FLOAT,
                                       [1, SEQ, aco.shape[0]])],
        inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="vv-head")
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, ONNX_PATH)
    del inits, model
    print("wrote", ONNX_PATH, os.path.getsize(ONNX_PATH) // 1048576, "MB", flush=True)


def main():
    cmd = sys.argv[1]
    if cmd == "export":
        export()
        return

    import qai_hub as hub
    c = hub.Client()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}

    if cmd == "submit":
        if "model" not in st:
            m = c.upload_model(ONNX_PATH, name="vv_head_fp16")
            st["model"] = m.model_id
            json.dump(st, open(STATE, "w"), indent=1)
            print("model", m.model_id, flush=True)
        dev = c.get_job("j57o7d8vg").device   # Samsung Galaxy S25 (Family)
        j = c.submit_compile_job(
            model=c.get_model(st["model"]), device=dev, name="vv_head_fp16",
            options="--target_runtime qnn_dlc --truncate_64bit_io --qairt_version 2.51")
        st["compile"] = j.job_id
        json.dump(st, open(STATE, "w"), indent=1)
        print("compile", j.job_id, flush=True)

    elif cmd == "poll":
        s = c.get_job(st["compile"]).get_status()
        print("compile", s.code, s.message[:200], flush=True)

    elif cmd == "link":
        j = c.get_job(st["compile"])
        if j.get_status().code != "SUCCESS":
            raise SystemExit("compile not done: " + j.get_status().code)
        lj = c.submit_link_job(models=j.get_target_model(), device=j.device,
                               name="vv_head_fp16_link",
                               options="--qairt_version 2.51")
        st["link"] = lj.job_id
        json.dump(st, open(STATE, "w"), indent=1)
        print("link", lj.job_id, flush=True)

    elif cmd == "poll-link":
        s = c.get_job(st["link"]).get_status()
        print("link", s.code, s.message[:200], flush=True)

    elif cmd == "fetch":
        j = c.get_job(st["link"])
        if j.get_status().code != "SUCCESS":
            raise SystemExit("link not done: " + j.get_status().code)
        dest = os.path.join(HERE, "head.bin")
        j.download_target_model(dest)
        print("saved", dest, os.path.getsize(dest), "B", flush=True)

    else:
        raise SystemExit("unknown command")


if __name__ == "__main__":
    main()
