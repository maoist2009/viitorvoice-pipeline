#!/usr/bin/env python3
"""Staticize dualcodec_encode_core_30s.onnx to the app's on-device window.

The app pads pcm24k to 192900 samples (= 8.0375 s @24 kHz) and feeds the
matching 200-frame semantic_repr from w2v_front8s.onnx, so the compile must
see exactly those shapes: audio (1,1,192900) + semantic_repr (1,1024,200).

usage: encoder_static_200.py [out_path]
"""
import sys

import onnx

SRC = ("/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/"
       "viitor-voice-nar/local_models/dualcodec/dualcodec_ckpts/"
       "dualcodec_encode_core_30s.onnx")
OUT = sys.argv[1] if len(sys.argv) > 1 else (
    "/data/data/com.termux/files/usr/tmp/encode_core_s200.onnx")
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 192900
N_FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 200
FIX = {
    "audio": (1, 1, N_SAMPLES),
    "semantic_repr": (1, 1024, N_FRAMES),
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

    bad = 0
    for vi in list(m.graph.value_info) + list(m.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if not dims:
            bad += 1
            continue
        for d in dims:
            if not d.HasField("dim_value"):
                bad += 1
                print("  still dynamic:", vi.name,
                      [x.dim_value if x.HasField("dim_value") else x.dim_param for x in dims],
                      flush=True)
                break
    print("dynamic tensors left:", bad, flush=True)
    print("outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                       for o in m.graph.output], flush=True)

    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
