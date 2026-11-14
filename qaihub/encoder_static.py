#!/usr/bin/env python3
"""Staticize the dualcodec encoder ONNX (dynamic batch / samples / frames).

qai-hub rejects the dynamic model when `--quantize_io` is off ("Model input
'audio' has dynamic shapes"), and we must turn `--quantize_io` off because its
heuristic [0,1] IO range clips this graph's signed inputs. So the shapes are
frozen here instead: batch=1, audio 240000 samples (10 s @24 kHz), 600 frames.

usage: encoder_static.py [out_path]
"""
import sys

import onnx

SRC = ("/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/"
       "viitor-voice-nar/local_models/dualcodec/dualcodec_ckpts/"
       "dualcodec_encode_core_30s.onnx")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/data/com.termux/files/usr/tmp/encode_static.onnx"
FIX = {
    "audio": (1, 1, 240000),
    "semantic_repr": (1, 1024, 600),
}


def staticize(t):
    dims = t.type.tensor_type.shape.dim
    want = FIX[t.name]
    if len(dims) != len(want):
        raise SystemExit(f"{t.name}: rank {len(dims)} != {len(want)}")
    for d, v in zip(dims, want):
        d.ClearField("dim_param")
        d.dim_value = v


def main():
    m = onnx.load(SRC)
    for t in m.graph.input:
        staticize(t)
    print("inputs fixed:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                            for i in m.graph.input], flush=True)
    try:
        m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    except Exception as e:                       # noqa: BLE001
        print("strict inference failed:", e, "-> retrying lax", flush=True)
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False)

    unknown = 0
    for vi in list(m.graph.value_info) + list(m.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if not dims:
            unknown += 1
            continue
        for d in dims:
            if not d.HasField("dim_value"):
                unknown += 1
                print("  still dynamic:", vi.name,
                      [x.dim_value if x.HasField("dim_value") else x.dim_param for x in dims],
                      flush=True)
                break
    print("dynamic tensors left:", unknown, flush=True)

    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
