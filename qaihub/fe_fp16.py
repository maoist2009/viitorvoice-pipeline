#!/usr/bin/env python3
"""w2v_front8s: valid_len(samples) -> valid_frames(mel frames), fp32 -> fp16.

Why the graph change: valid_len carries the sample count, e.g. 8.04 s * 16 kHz =
128640, which fp16 cannot represent (max 65504) and integers above 2048 are not
exact. The graph only uses it as floor((v - 400) / 160) + 1, so feed the mel
frame count directly (<= 798, exact in fp16).

Why fp16: the fp32 graph is 2.25 GB (external data) which exceeds the protobuf
2 GiB single-message limit; fp16 gives a ~1.1 GB single file that uploads whole.

The conversion is all-or-nothing on purpose: onnxconverter_common leaves
inconsistent Cast/value_info types on this graph, so every FLOAT here becomes
FLOAT16 (initializers, graph IO, Cast targets) and stale value_info is dropped.
That is what QAIRT's --quantize_full_type float16 does anyway.

usage: fe_fp16.py <src.onnx> <dst.onnx>
"""
import gc
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

src, dst = sys.argv[1], sys.argv[2]
mid = os.path.join(os.path.dirname(os.path.abspath(src)),
                   os.path.splitext(os.path.basename(src))[0] + "_vf.onnx")

# ---------------------------------------------------------------- pass 1: surgery
m = onnx.load(src, load_external_data=False)
g = m.graph

assert any(i.name == "valid_len" for i in g.input), "valid_len input missing"
keep = [i for i in g.input if i.name != "valid_len"]
del g.input[:]
g.input.extend(keep)
g.input.append(helper.make_tensor_value_info("valid_frames", TensorProto.FLOAT, [1]))

# floor((valid_len - 400) / 160) + 1  ==  valid_frames
drop = {"sub_2", "div", "floor", "add"}
seen, nodes = [], []
for n in g.node:
    out = set(n.output)
    if out & drop:
        seen.append(n.op_type)
        if "add" in out:               # keep topological position
            # float (not int32) so no Cast is needed: frame counts <= 798 are
            # exact in fp16
            nodes.append(helper.make_node("Identity", ["valid_frames"], ["add"]))
        continue
    nodes.append(n)
assert sorted(seen) == ["Add", "Div", "Floor", "Sub"], seen
del g.node[:]
g.node.extend(nodes)
onnx.save(m, mid)                      # weights stay external, untouched
print("surgery ->", mid, "inputs:", [i.name for i in g.input])
del m, g
gc.collect()

# ------------------------------------------------------- pass 2: uniform fp16
m = onnx.load(mid)                     # pulls the 2.25 GB .data in
g = m.graph

clipped = 0
for t in g.initializer:
    if t.data_type != TensorProto.FLOAT:
        continue
    a = np.frombuffer(t.raw_data, dtype=np.float32)
    if a.size and float(np.abs(a).max()) > 65504.0:
        clipped += 1
        a = np.clip(a, -65504.0, 65504.0)   # -FLT_MAX used as a mask value
    t.data_type = TensorProto.FLOAT16
    t.raw_data = np.asarray(a, dtype=np.float16).tobytes()

for coll in (g.input, g.output):
    for vi in coll:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16

n_cast = 0
for n in g.node:
    if n.op_type == "Cast":
        for a in n.attribute:
            if a.name == "to" and a.i == TensorProto.FLOAT:
                a.i = TensorProto.FLOAT16
                n_cast += 1

del g.value_info[:]                    # stale fp32 declarations
gc.collect()
onnx.save(m, dst)
sz = os.path.getsize(dst)
print(f"fp16: clipped_consts={clipped} cast_fixed={n_cast} size={sz} B ({sz / 2**30:.2f} GiB)")
