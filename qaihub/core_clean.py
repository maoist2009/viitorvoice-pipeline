#!/usr/bin/env python3
"""Fully staticize + clean a dualcodec encode-core ONNX.

The graph that torch.export produced for TensorRT carries a Shape/Gather/
ConstantOfShape machinery that keeps 500+ tensors shapeless.  QAIRT cannot
resolve those and falls back to a fixed tensor size, which then overflows HTP
VTCM.  This script:

  1. fixes input shapes to (1,1,N) / (1,1024,F),
  2. probes the graph with onnxruntime to learn the value of every tensor that
     shape inference cannot resolve (two different inputs, so anything that is
     data dependent is detected and left alone),
  3. bakes genuinely constant tensors as initializers and drops their producers,
  4. repeats until no tensor is shapeless,
  5. dead-code eliminates, drops unused initializers,
  6. verifies the result numerically against the input model.

usage: core_clean.py <src.onnx> <samples> <frames> <dst.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort

SRC = sys.argv[1]
N = int(sys.argv[2])
F = int(sys.argv[3])
DST = sys.argv[4]
FIX = {"audio": (1, 1, N), "semantic_repr": (1, 1024, F)}

TNAME = {1: "f32", 10: "f16", 11: "f64", 7: "i64", 6: "i32", 9: "bool", 2: "u8"}


def log(m):
    print(m, flush=True)


def fix_inputs(m):
    for t in m.graph.input:
        want = FIX[t.name]
        for d, v in zip(t.type.tensor_type.shape.dim, want):
            d.ClearField("dim_param")
            d.dim_value = v


def infer(m):
    try:
        return onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    except Exception as e:                              # noqa: BLE001
        log(f"  infer_shapes failed: {str(e)[:120]}")
        return m


def tensor_map(m):
    out = {}
    for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
        out[vi.name] = vi.type.tensor_type
    for t in m.graph.initializer:
        out[t.name] = None
    return out


def unknown_tensors(m):
    """(name, elem_type) of tensors with no fully known shape."""
    res = []
    for vi in list(m.graph.value_info) + list(m.graph.output):
        tt = vi.type.tensor_type
        dims = tt.shape.dim
        if not dims or any(not d.HasField("dim_value") for d in dims):
            res.append((vi.name, tt.elem_type))
    return res


def make_probe(m, names, types):
    p = onnx.ModelProto()
    p.CopyFrom(m)
    have = {o.name for o in p.graph.output}
    for nm, et in zip(names, types):
        if nm in have:
            continue
        p.graph.output.append(helper.make_tensor_value_info(nm, et, None))
    return p


def run_probe(m, names, types, feeds):
    p = make_probe(m, names, types)
    sess = ort.InferenceSession(p.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    return sess.run(names, feeds)


def feeds_for(seed):
    rng = np.random.default_rng(seed)
    return {
        "audio": rng.standard_normal((1, 1, N), dtype=np.float32) * 0.1,
        "semantic_repr": rng.standard_normal((1, 1024, F), dtype=np.float32) * 0.1,
    }


def probe_all(m, cand, feeds):
    """values for the candidate tensors in one ORT session."""
    names = [c[0] for c in cand]
    types = [c[1] if c[1] else TensorProto.FLOAT for c in cand]
    return dict(zip(names, run_probe(m, names, types, feeds)))


def set_shape(m, name, dims):
    tt = None
    for vi in list(m.graph.value_info) + list(m.graph.output):
        if vi.name == name:
            tt = vi.type.tensor_type
            break
    if tt is None:
        return False
    tt.shape.ClearField("dim")
    for d in dims:
        tt.shape.dim.add().dim_value = int(d)
    return True


def prune(m):
    used = {n for node in m.graph.node for n in node.input if n}
    used |= {o.name for o in m.graph.output}
    keep = [n for n in m.graph.node if any(o in used for o in n.output if o)]
    while True:
        produced = {o for n in keep for o in n.output if o}
        need = {n for node in keep for n in node.input if n}
        dropped = [n for n in keep
                   if not any(o in need or o in {x.name for x in m.graph.output}
                              for o in n.output if o)]
        if not dropped:
            break
        keep = [n for n in keep if n not in dropped]
        produced = {o for n in keep for o in n.output if o}
        log(f"  prune round: -{len(dropped)}")
    m.graph.ClearField("node")
    m.graph.node.extend(keep)
    have = {i.name for i in m.graph.initializer}
    still = {n for node in m.graph.node for n in node.input if n}
    outs = {o.name for o in m.graph.output}
    keepi = [i for i in m.graph.initializer if i.name in still or i.name in outs]
    if len(keepi) != len(m.graph.initializer):
        log(f"  drop initializers {len(m.graph.initializer) - len(keepi)}")
        m.graph.ClearField("initializer")
        m.graph.initializer.extend(keepi)
    del have


def purge_value_info(m):
    """drop value_info entries that are initializers or no longer produced."""
    inits = {i.name for i in m.graph.initializer}
    produced = {o for n in m.graph.node for o in n.output if o}
    keep = [vi for vi in m.graph.value_info
            if vi.name not in inits and vi.name in produced]
    if len(keep) != len(m.graph.value_info):
        m.graph.ClearField("value_info")
        m.graph.value_info.extend(keep)


def bake(m, values):
    """turn tensor values into initializers, remove their producers."""
    have = {i.name for i in m.graph.initializer}
    producers = {}
    for node in m.graph.node:
        for o in node.output:
            producers[o] = node
    kill = set()
    added = 0
    for name, val in values.items():
        if name in have:
            continue
        m.graph.initializer.append(numpy_helper.from_array(np.asarray(val), name))
        added += 1
        node = producers.get(name)
        if node is not None:
            kill.add(id(node))
    keep = [n for n in m.graph.node if id(n) not in kill]
    m.graph.ClearField("node")
    m.graph.node.extend(keep)
    purge_value_info(m)
    return added


def shape_of(m):
    """tensor -> dims for everything with a fully known shape."""
    out = {i.name: tuple(i.dims) for i in m.graph.initializer}
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
            out[vi.name] = tuple(vals)
    return out


def eval_node(node, known):
    from onnx.reference import ReferenceEvaluator
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
    mdl = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    mdl.ir_version = 8
    return ReferenceEvaluator(mdl).run(None, feeds)


def fold_constants(m, max_elems=1 << 20):
    """fold Shape-of-static and any node whose inputs are all constants."""
    known = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    shapes = shape_of(m)
    kill = set()
    folded = 0
    for node in m.graph.node:
        if node.op_type == "Constant":
            continue
        if node.op_type == "Shape" and node.input[0] in shapes:
            known[node.output[0]] = np.array(shapes[node.input[0]], dtype=np.int64)
            kill.add(id(node))
            folded += 1
            continue
        ins = [n for n in node.input if n]
        if not ins or not all(n in known for n in ins):
            continue
        try:
            vals = eval_node(node, known)
        except Exception:                               # noqa: BLE001
            continue
        if any(np.asarray(v).size > max_elems for v in vals):
            continue
        for nm, v in zip(node.output, vals):
            known[nm] = np.asarray(v)
        kill.add(id(node))
        folded += 1
    keep = [n for n in m.graph.node if id(n) not in kill]
    m.graph.ClearField("node")
    m.graph.node.extend(keep)
    purge_value_info(m)
    return folded, known


def verify(src, dst):
    a = ort.InferenceSession(src.SerializeToString(), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(dst.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = feeds_for(7)
    oa = a.run(None, feeds)
    ob = b.run(None, feeds)
    names = [o.name for o in a.get_outputs()]
    worst = 0.0
    for i, (x, y) in enumerate(zip(oa, ob)):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            log(f"  MISMATCH shape out[{i}] {x.shape} vs {y.shape}")
            return False
        d = float(np.abs(x - y).max()) if x.size else 0.0
        worst = max(worst, d)
        log(f"  out[{i}] {names[i]} shape={x.shape} maxdiff={d:.3e}")
    return worst < 1e-5


def attach(m, known):
    have = {i.name for i in m.graph.initializer}
    produced = {o for n in m.graph.node for o in n.output if o}
    used = {n for node in m.graph.node for n in node.input if n}
    added = 0
    for name, val in known.items():
        if name in have or name in produced or name not in used:
            continue
        m.graph.initializer.append(numpy_helper.from_array(np.asarray(val), name))
        added += 1
    return added


def main():
    m = onnx.load(SRC)
    fix_inputs(m)
    log(f"loaded {SRC}: nodes={len(m.graph.node)} inits={len(m.graph.initializer)}")

    for rnd in range(12):
        m = infer(m)
        unk = unknown_tensors(m)
        log(f"round {rnd}: nodes={len(m.graph.node)} "
            f"inits={len(m.graph.initializer)} unknown={len(unk)}")
        if unk:
            v1 = probe_all(m, unk, feeds_for(1))
            v2 = probe_all(m, unk, feeds_for(2))
            same = {k: v1[k] for k in v1
                    if v1[k].shape == v2[k].shape and np.array_equal(v1[k], v2[k])}
            log(f"  probed {len(unk)}, constant {len(same)}, "
                f"data-dependent {len(unk) - len(same)}")
            baked = bake(m, same)
            fixed = sum(1 for k in v1 if k not in same and set_shape(m, k, v1[k].shape))
            log(f"  baked {baked} constants, annotated {fixed} shapes")
            if baked == 0 and fixed == 0:
                log("  stuck at probe stage")
                break
            m = infer(m)
        folded, known = fold_constants(m)
        added = attach(m, known)
        log(f"  folded {folded} nodes, attached {added} constants")
        if folded == 0 and added == 0:
            break

    prune(m)
    m = infer(m)
    unk = unknown_tensors(m)
    log(f"after clean: nodes={len(m.graph.node)} inits={len(m.graph.initializer)} "
        f"unknown={len(unk)}")
    for nm, et in unk[:10]:
        log(f"   still unknown: {nm} ({TNAME.get(et, et)})")
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, DST)
    log(f"saved {DST}")

    log("verify:")
    ok = verify(onnx.load(SRC), m)
    log("IDENTICAL" if ok else "DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
