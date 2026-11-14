#!/usr/bin/env python3
"""Split the time-sliced front into a *single-tile* graph + a block.3 graph.

HTP mis-computes ~1/3 of every tile when the graph holds eight clones of the
same block.0..2 subgraph (deterministic, input independent, ~90 % relative error
in those windows).  Running one tile per graph and concatenating on the host
removes the cloning from the compiled graph entirely.

produces
  front_tile.onnx : audio[1,1,24960] -> front[1,256,1200]   (halo trimmed)
  e4b.onnx        : front[1,256,9600] -> block.3 outputs (2 x [1,512,1600])
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SRC = sys.argv[1] if len(sys.argv) > 1 else "/data/data/com.termux/files/usr/tmp/enc_core_192000_clean.onnx"
OUT = "/data/data/com.termux/files/usr/tmp"
TILE = int(sys.argv[2]) if len(sys.argv) > 2 else 24000
HALO = int(sys.argv[3]) if len(sys.argv) > 3 else 480
NAME = sys.argv[4] if len(sys.argv) > 4 else "front_tile"
STRIDE = 20
FRONT_OUT = "/encoder/block/block.2/block/block.4/Conv_output_0"
B3_CONV = "/encoder/block/block.3/block/block.4/Conv_output_0"
B3_RES = "input.164"

m = onnx.load(SRC)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
inits = {i.name: i for i in g.initializer}
ginputs = {i.name for i in g.input}


def reach_back(targets, stop=()):
    need, seen = set(), set()
    stack = list(targets)
    while stack:
        t = stack.pop()
        if t in seen or t in inits or t in ginputs or t in stop:
            continue
        seen.add(t)
        n = prod.get(t)
        if n is None:
            raise SystemExit(f"no producer for {t}")
        if n.name == "/Pad":
            continue
        need.add(n.name)
        stack.extend(i for i in n.input if i)
    return need


# ---------------------------------------------------------------- front tile
front_names = reach_back([FRONT_OUT])
front = [n for n in g.node if n.name in front_names]

tlen = TILE + 2 * HALO
frames = TILE // STRIDE
halo_frames = HALO // STRIDE

scaled = {}
for nm in {i for n in front for i in n.input if i}:
    ini = inits.get(nm)
    if ini is None or ini.data_type != TensorProto.INT64:
        continue
    a = numpy_helper.to_array(ini)
    if a.shape == (3,) and a[0] == 1 and a[1] > 1 and a[2] > 1:
        num = int(a[2]) * tlen
        assert num % 192000 == 0, (nm, a.tolist())
        scaled[nm] = np.array([1, int(a[1]), num // 192000], np.int64)

nodes = []
for n in front:
    nn = onnx.NodeProto()
    nn.CopyFrom(n)
    nn.input[:] = [("audio" if t == "/Pad_output_0" else t) for t in nn.input]
    nodes.append(nn)


used = {i for n in nodes for i in n.input if i}
kept = [numpy_helper.from_array(scaled[nm], nm) if nm in scaled else inits[nm]
        for nm in sorted(used) if nm in inits]
ng = helper.make_graph(
    nodes, "front_tile",
    [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, tlen])],
    [helper.make_tensor_value_info(FRONT_OUT, TensorProto.FLOAT,
                                   [1, 256, frames + 2 * halo_frames])],
    kept)
fm = helper.make_model(ng, opset_imports=list(m.opset_import))
fm.ir_version = m.ir_version
onnx.checker.check_model(fm, full_check=True)
onnx.save(fm, f"{OUT}/{NAME}.onnx")
print(f"{NAME}: {len(nodes)} nodes, {tlen} samples -> {frames + 2 * halo_frames} frames (untrimmed)")

# -------------------------------------------------------------------- e4b
b_names = reach_back([B3_CONV, B3_RES], stop={FRONT_OUT})
bnodes = [n for n in g.node if n.name in b_names]
used = {i for n in bnodes for i in n.input if i}
kept = [inits[nm] for nm in sorted(used) if nm in inits]
bg = helper.make_graph(
    bnodes, "e4b",
    [helper.make_tensor_value_info(FRONT_OUT, TensorProto.FLOAT,
                                   [1, 256, (192000 // TILE) * frames])],
    [helper.make_tensor_value_info(B3_CONV, TensorProto.FLOAT, [1, 512, 1600]),
     helper.make_tensor_value_info(B3_RES, TensorProto.FLOAT, [1, 512, 1600])],
    kept)
bm = helper.make_model(bg, opset_imports=list(m.opset_import))
bm.ir_version = m.ir_version
onnx.checker.check_model(bm, full_check=True)
onnx.save(bm, f"{OUT}/e4b.onnx")
print(f"e4b: {len(bnodes)} nodes, {8 * frames} frames -> 2 x 1600")
