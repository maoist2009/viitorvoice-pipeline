#!/usr/bin/env python3
"""Fully staticize dualcodec_encode_core_30s.onnx by constant-folding the
Shape/Gather/Concat/Reshape machinery that torch's dynamic export leaves in.

QAIRT cannot resolve those computed reshapes and falls back to a fixed tensor
size, which then overflows HTP VTCM ("activations_from_vtcm requires 0xffb000
bytes" > 0x800000).  Folding the shape chain to constants makes every tensor
statically shaped.

usage: core_fold.py <out.onnx> [samples] [frames]
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

SRC = (sys.argv[4] if len(sys.argv) > 4 else
       "/data/data/com.termux/files/home/projects/voice_editor/sm8750_voice_editor/"
       "viitor-voice-nar/local_models/dualcodec/dualcodec_ckpts/"
       "dualcodec_encode_core_30s.onnx")
OUT = sys.argv[1] if len(sys.argv) > 1 else (
    "/data/data/com.termux/files/usr/tmp/encode_core_static.onnx")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 192900
F = int(sys.argv[3]) if len(sys.argv) > 3 else 200
FIX = {"audio": (1, 1, N), "semantic_repr": (1, 1024, F)}
OPSET = 18


def log(m):
    print(m, flush=True)


def staticize_inputs(m):
    for t in m.graph.input:
        want = FIX[t.name]
        dims = t.type.tensor_type.shape.dim
        for d, v in zip(dims, want):
            d.ClearField("dim_param")
            d.dim_value = v


def eval_node(node, known):
    ins, feeds = [], {}
    for name in node.input:
        if not name:
            continue
        arr = known[name]
        ins.append(helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(arr.dtype), list(arr.shape)))
        feeds[name] = arr
    outs = [helper.make_tensor_value_info(o, TensorProto.FLOAT, None)
            for o in node.output]
    g = helper.make_graph([node], "g", ins, outs)
    mdl = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    mdl.ir_version = 8
    res = ReferenceEvaluator(mdl).run(None, feeds)
    return res


def collect_shapes(m, shapes):
    for init in m.graph.initializer:
        shapes[init.name] = tuple(init.dims)
    for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if not dims:
            continue
        vals = []
        for d in dims:
            if not d.HasField("dim_value"):
                vals = None
                break
            vals.append(d.dim_value)
        if vals is not None:
            shapes[vi.name] = tuple(vals)


def main():
    m = onnx.load(SRC)
    staticize_inputs(m)
    m.graph.ClearField("value_info")
    known = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    log(f"initializers {len(known)}")

    for it in range(8):
        try:
            m = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
        except Exception as e:                          # noqa: BLE001
            log(f"  infer failed: {str(e)[:120]}")
        shapes = {}
        collect_shapes(m, shapes)
        changed = False
        keep = []
        for node in m.graph.node:
            if node.op_type == "Constant":
                for a in node.attribute:
                    if a.name == "value":
                        known[node.output[0]] = numpy_helper.to_array(a.t)
                continue
            if node.op_type == "Shape" and node.input[0] in shapes:
                known[node.output[0]] = np.array(shapes[node.input[0]], dtype=np.int64)
                changed = True
                continue
            if all(n in known for n in node.input if n):
                try:
                    vals = eval_node(node, known)
                    for nm, v in zip(node.output, vals):
                        known[nm] = np.asarray(v)
                    changed = True
                    continue
                except Exception as e:                  # noqa: BLE001
                    if it == 0:
                        log(f"  fold skip {node.op_type} {node.name}: {str(e)[:80]}")
            keep.append(node)
        m.graph.ClearField("node")
        m.graph.node.extend(keep)
        log(f"pass {it}: nodes {len(keep)} constants {len(known)} changed {changed}")
        if not changed:
            break
        m.graph.ClearField("value_info")

    # attach constants that surviving nodes still consume
    have = {i.name for i in m.graph.initializer}
    used = {n for node in m.graph.node for n in node.input if n}
    produced = {n for node in m.graph.node for n in node.output if n}
    added = 0
    for name in used:
        if name in known and name not in have and name not in produced:
            arr = known[name]
            m.graph.initializer.append(numpy_helper.from_array(arr, name))
            added += 1
    log(f"added initializers {added}")

    shapes = {}
    collect_shapes(m, shapes)
    dyn = 0
    for vi in list(m.graph.value_info) + list(m.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if not dims or any(not d.HasField("dim_value") for d in dims):
            dyn += 1
    log(f"dynamic tensors left {dyn}")
    log("outputs " + str([(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                          for o in m.graph.output]))
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, OUT)
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
