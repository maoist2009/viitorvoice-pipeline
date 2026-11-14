#!/usr/bin/env python3
"""Retarget the dualcodec encode-core to a shorter maximum audio window.

The exported graph always pads the audio to ``max_samples`` (a Constant node,
720000 = 30 s @ 24 kHz) and then runs the full encoder over it.  That makes the
first conv activation [1,64,720000] (~184 MB) which cannot be tiled into the
8 MB HTP VTCM ("activations_from_vtcm requires 0xffb000 bytes of TCM").

Rewriting that one constant to a shorter window makes every downstream shape
(shapes are derived from it via Shape/Concat arithmetic) follow automatically.

usage: core_pad_patch.py <out.onnx> <samples>
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper

SRC = ("/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/"
       "viitor-voice-nar/local_models/dualcodec/dualcodec_ckpts/"
       "dualcodec_encode_core_30s.onnx")
OUT = sys.argv[1]
N = int(sys.argv[2])

m = onnx.load(SRC)
patched = 0
for node in m.graph.node:
    if node.op_type != "Constant":
        continue
    for a in node.attribute:
        if a.name != "value":
            continue
        arr = numpy_helper.to_array(a.t)
        if arr.size == 1 and int(arr.reshape(-1)[0]) == 720000:
            a.t.CopyFrom(numpy_helper.from_array(np.array(N, dtype=arr.dtype)))
            patched += 1
            print(f"patched {node.output[0]} -> {N}")
print("patched nodes:", patched)
onnx.save(m, OUT)
print("saved", OUT)
