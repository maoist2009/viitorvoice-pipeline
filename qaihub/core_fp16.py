#!/usr/bin/env python3
"""Cast a fully-static float32 encode-core graph to float16.

QNN's HTP graph prepare sizes the VTCM tile of the first full-rate Snake
activation from the *fp32* activation bytes, which for an 8 s window is
~16.4 MB > the 8 MB VTCM and the context-binary link fails.  Feeding the
compiler an fp16 graph halves that estimate.

The graph is homogeneous fp32 at this point (all weight-normalisation and
shape arithmetic were constant-folded), so a plain dtype swap is enough: the
int64 index tensors and the int code outputs are left alone.

usage: core_fp16.py <in.onnx> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

SRC, OUT = sys.argv[1], sys.argv[2]
m = onnx.load(SRC)

converted = 0
inits = []
for init in m.graph.initializer:
    if init.data_type == TensorProto.FLOAT:
        arr = numpy_helper.to_array(init).astype(np.float16)
        inits.append(numpy_helper.from_array(arr, init.name))
        converted += 1
    else:
        inits.append(init)
m.graph.ClearField("initializer")
m.graph.initializer.extend(inits)

nodes = []
for node in m.graph.node:
    if node.op_type == "Constant":
        for a in node.attribute:
            if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(a.t).astype(np.float16)
                a.t.CopyFrom(numpy_helper.from_array(arr))
                converted += 1
    elif node.op_type == "ConstantOfShape":
        for a in node.attribute:
            if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(a.t).astype(np.float16)
                a.t.CopyFrom(numpy_helper.from_array(arr))
                converted += 1
    elif node.op_type == "Cast":
        for a in node.attribute:
            if a.name == "to" and a.i == TensorProto.FLOAT:
                a.i = TensorProto.FLOAT16
                converted += 1
    elif node.op_type in ("RandomNormal", "RandomUniform"):
        raise SystemExit(f"{node.op_type} would need manual dtype handling")
    nodes.append(node)

for vi in list(m.graph.input) + list(m.graph.output):
    if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
        vi.type.tensor_type.elem_type = TensorProto.FLOAT16

m.graph.ClearField("value_info")
onnx.checker.check_model(m, full_check=True)
onnx.save(m, OUT)
import os
print(f"converted {converted} fp32 tensors; {os.path.getsize(OUT)/1e6:.1f} MB -> {OUT}")
print("inputs ", [(i.name, i.type.tensor_type.elem_type) for i in m.graph.input])
print("outputs", [(o.name, o.type.tensor_type.elem_type) for o in m.graph.output])
