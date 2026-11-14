#!/usr/bin/env python3
"""Split the cleaned encode-core graph into 6 chained segments.

Motivation: qai-hub's multipart upload stalls past ~80 MB and the model is
197 MB, so it can only be uploaded piecemeal.  The cut points follow the
natural stage boundaries of the network, which also happens to be how the
decoder already ships (dec_p1..p5).

  e1  semantic_repr -> convnext_encoder.0-3        ~41 MB
  e2  convnext_encoder.4 + semantic_vq + decoder.0-1  ~39 MB
  e3  convnext_decoder.2-4                        ~28 MB
  e4  /encoder/block.0-3  (+ audio pad)           ~16 MB
  e5  /encoder/block.4                            ~59 MB
  e6  /encoder/block.5-6 + RVQ                    ~15 MB

usage: core_split.py <src.onnx> <outdir>
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto

SRC, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
TNAME = {1: "f32", 10: "f16", 11: "f64", 7: "i64", 6: "i32", 9: "bool"}


def level_of(name):
    import re
    if name == "/Pad" or name.startswith("/slice/"):
        return 4                       # audio pad / time slicing: audio path
    if name.startswith("/convnext_encoder/convnext_encoder."):
        return 1 if int(re.match(r"/convnext_encoder/convnext_encoder\.(\d+)", name).group(1)) <= 3 else 2
    if name.startswith("/semantic_vq/"):
        return 2
    if name.startswith("/convnext_decoder/convnext_decoder."):
        k = int(re.match(r"/convnext_decoder/convnext_decoder\.(\d+)", name).group(1))
        return 2 if k <= 1 else 3
    if name.startswith("/encoder/block/block."):
        k = int(re.match(r"/encoder/block/block\.(\d+)", name).group(1))
        return 4 if k <= 3 else (5 if k == 4 else 6)
    if name.startswith("/quantizer/"):
        return 6
    return None


def main():
    m = onnx.load(SRC, load_external_data=False)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    lvl = {}
    consumers = {}
    for n in g.node:
        own = level_of(n.name)
        dep = max([lvl.get(i, 0) for i in n.input if i] or [0])
        lv = dep if own is None else own
        assert dep <= lv, f"level violation at {n.name}: dep {dep} own {own}"
        for o in n.output:
            lvl[o] = lv
        for i in n.input:
            if i:
                consumers.setdefault(i, []).append(lv)
    for i in g.input:
        lvl[i.name] = 0
        consumers.setdefault(i.name, []).append(0)

    groups = {1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]}
    tag = {1: "e1", 2: "e2", 3: "e3", 4: "e4", 5: "e5", 6: "e6"}
    inits = {i.name: i for i in g.initializer}
    vinfo = {vi.name: vi for vi in list(g.input) + list(g.value_info) + list(g.output)}
    gout = {o.name for o in g.output}

    for k, levels in groups.items():
        nodes = [n for n in g.node if lvl.get(n.output[0], 0) in levels]
        produced = {o for n in nodes for o in n.output}
        used = {i for n in nodes for i in n.input if i}
        gin = [i for i in g.input if i.name in used]
        ext_in = sorted(t for t in used if t in lvl and t not in produced
                        and t not in inits and t not in {x.name for x in gin})
        out_ts = sorted(t for t in produced
                        if t in gout or max(consumers.get(t, [0])) > max(levels))
        need_init = [inits[t] for t in sorted(used) if t in inits]
        graph = onnx.GraphProto()
        graph.name = tag[k]
        for t in ext_in + [x.name for x in gin]:
            vi = onnx.ValueInfoProto()
            vi.CopyFrom(_vi(vinfo[t], t))
            graph.input.append(vi)
        for t in out_ts:
            vi = onnx.ValueInfoProto()
            vi.CopyFrom(_vi(vinfo[t], t, rank_only=False))
            graph.output.append(vi)
        graph.node.extend(nodes)
        graph.initializer.extend(need_init)
        mdl = onnx.ModelProto()
        mdl.ir_version = m.ir_version
        mdl.opset_import.extend(m.opset_import)
        mdl.graph.CopyFrom(graph)
        path = os.path.join(OUTDIR, f"{tag[k]}.onnx")
        onnx.checker.check_model(mdl, full_check=True)
        onnx.save(mdl, path)
        wb = sum(len(i.raw_data) for i in need_init) / 1e6
        print(f"{tag[k]}: nodes={len(nodes):3d} inits={len(need_init):3d} "
              f"w={wb:5.1f}MB in={[x.name for x in gin] + ext_in} "
              f"out={[(t, _dims(vinfo[t])) for t in out_ts]} "
              f"file={os.path.getsize(path)/1e6:.1f}MB", flush=True)


def _dims(vi):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in vi.type.tensor_type.shape.dim]


def _vi(src, name, rank_only=False):
    vi = onnx.ValueInfoProto()
    vi.name = name
    vi.type.tensor_type.elem_type = src.type.tensor_type.elem_type
    if not rank_only:
        for d in src.type.tensor_type.shape.dim:
            nd = vi.type.tensor_type.shape.dim.add()
            if d.HasField("dim_value"):
                nd.dim_value = d.dim_value
            else:
                nd.dim_param = d.dim_param or "?"
    return vi


if __name__ == "__main__":
    main()
