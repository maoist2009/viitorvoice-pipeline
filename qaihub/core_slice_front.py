#!/usr/bin/env python3
"""Make the full-rate front of the encode core VTCM-friendly by time slicing.

HTP refuses to tile the first two encoder blocks: it wants the whole 64-channel
activation in VTCM (192000 samples -> 16.4 MB, 46080 -> 11.8 MB, limit 8 MB).
The window would have to shrink to ~34 frames, but the encoder's receptive
field is only 439 input samples, so slicing *inside* the graph is exact and
cheap.

This rewrites the graph as

    audio[1,1,192000]
      -> Pad(480 both ends)  -> 8 x Slice(24960)  -> 8 clones of block.0..2
      -> Slice(24..1224) each -> Concat -> 9600 frames
      -> block.3 (unmodified)

Halo 480 >= receptive field 439 (6 + 78 + 7 + 312 + 36), and both 480 and the
tile length are multiples of 20 so the stride-5 boundary stays aligned.

usage: core_slice_front.py <img.onnx> <dst.onnx> [tile halo]
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SRC, DST = sys.argv[1], sys.argv[2]
TILE = int(sys.argv[3]) if len(sys.argv) > 3 else 24000
HALO = int(sys.argv[4]) if len(sys.argv) > 4 else 480
AUDIO = "audio"
FRONT_OUT = "/encoder/block/block.2/block/block.4/Conv_output_0"
STRIDE = 20          # block.2 output rate: 192000 -> 9600 frames


def main():
    m = onnx.load(SRC, load_external_data=False)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    inits = {i.name: i for i in g.initializer}
    ginputs = {i.name for i in g.input}

    assert 192000 % TILE == 0 and TILE % STRIDE == 0 and HALO % STRIDE == 0
    tiles = 192000 // TILE
    frames_per_tile = TILE // STRIDE
    halo_frames = HALO // STRIDE

    # ---- split the graph: front (audio -> FRONT_OUT) vs the rest -------------
    def reach_back(targets):
        need, seen = set(), set()
        stack = list(targets)
        while stack:
            t = stack.pop()
            if t in seen or t in inits or t in ginputs:
                continue
            seen.add(t)
            n = prod.get(t)
            if n is None:
                raise SystemExit(f"no producer for {t}")
            if n.name == "/Pad":            # no-op zero pad, rebuilt below
                continue
            need.add(n.name)
            stack.extend(i for i in n.input if i)
        return need

    front_names = reach_back([FRONT_OUT])
    front = [n for n in g.node if n.name in front_names]
    rest = _prune_dead([n for n in g.node if n.name not in front_names], g)

    # "/Pad" is dropped (it pads zero samples), so its output becomes `audio`
    SUBST = {"/Pad_output_0": "__TILE__"}
    ext = {(AUDIO if i == "/Pad_output_0" else i) for n in front for i in n.input if i}
    ext -= {o for n in front for o in n.output} | set(inits)
    assert ext == {AUDIO}, f"front reads more than audio: {ext}"
    front_prod = {o for n in front for o in n.output}
    rest_in = {i for n in rest for i in n.input if i}
    cross = (front_prod & rest_in) - {FRONT_OUT}
    assert not cross, f"front also feeds the rest: {cross}"

    # ---- build ---------------------------------------------------------------
    nodes = [helper.make_node("Pad", [AUDIO, "c_pads"], ["audio_padded"],
                              mode="constant", name="/slice/Pad")]
    consts = [numpy_helper.from_array(np.array([0, 0, HALO, 0, 0, HALO], np.int64),
                                      "c_pads")]

    # Snake (and friends) reshape back with a *baked* length, e.g. [1,64,192000].
    # Every tile needs its own scaled copy of those shape constants.
    scaled = {}
    for nm, ini in inits.items():
        if ini.data_type != TensorProto.INT64:
            continue
        a = numpy_helper.to_array(ini)
        if a.shape == (3,) and a[0] == 1 and a[1] > 1 and a[2] > 1:
            scaled[nm] = a
    tlen = TILE + 2 * HALO

    for i in range(tiles):
        st = i * TILE
        for nm, val in ((f"c_s{i}", st), (f"c_e{i}", st + TILE + 2 * HALO)):
            consts.append(numpy_helper.from_array(np.array([val], np.int64), nm))
        consts.append(numpy_helper.from_array(np.array([2], np.int64), f"c_a{i}"))
        nodes.append(helper.make_node(
            "Slice", ["audio_padded", f"c_s{i}", f"c_e{i}", f"c_a{i}"],
            [f"tile{i}_in"], name=f"/slice/in{i}"))

        for nm, a in scaled.items():
            num = int(a[2]) * tlen
            assert num % 192000 == 0, (nm, a.tolist(), tlen)
            consts.append(numpy_helper.from_array(
                np.array([1, int(a[1]), num // 192000], np.int64), f"{nm}__t{i}"))

        for n in front:
            nn = onnx.NodeProto()
            nn.CopyFrom(n)
            nn.name = f"{n.name}__t{i}"
            nn.input[:] = [_rename(t, i, inits, scaled) for t in nn.input]
            nn.output[:] = [f"{t}__t{i}" if t else t for t in nn.output]
            nodes.append(nn)

        for nm, val in ((f"t_s{i}", halo_frames), (f"t_e{i}", halo_frames + frames_per_tile)):
            consts.append(numpy_helper.from_array(np.array([val], np.int64), nm))
        consts.append(numpy_helper.from_array(np.array([2], np.int64), f"t_a{i}"))
        nodes.append(helper.make_node(
            "Slice", [f"{FRONT_OUT}__t{i}", f"t_s{i}", f"t_e{i}", f"t_a{i}"],
            [f"tile{i}_out"], name=f"/slice/trim{i}"))

    nodes.append(helper.make_node(
        "Concat", [f"tile{i}_out" for i in range(tiles)], [FRONT_OUT],
        axis=2, name="/slice/Concat"))
    nodes.extend(rest)

    outputs = list(g.output)
    if os.environ.get("FRONT_ONLY"):
        outputs = [helper.make_tensor_value_info(FRONT_OUT, TensorProto.FLOAT,
                                                 [1, 256, 192000 // STRIDE])]
    used = {i for n in nodes for i in n.input if i}
    kept_inits = [inits[nm] for nm in sorted(used) if nm in inits]
    ng = helper.make_graph(nodes, "enc_front_sliced", list(g.input), outputs,
                           kept_inits + consts)
    alive = {o for n in nodes for o in n.output} | {o.name for o in g.output}
    ng.value_info.extend(vi for vi in g.value_info if vi.name in alive)
    nm_out = helper.make_model(ng, opset_imports=list(m.opset_import))
    nm_out.ir_version = m.ir_version
    try:
        onnx.checker.check_model(nm_out, full_check=True)
    except Exception as e:                              # noqa: BLE001
        print(f"  checker warning: {str(e).splitlines()[0][:110]}")
    onnx.save(nm_out, DST)
    print(f"tiles={tiles} tile={TILE} halo={HALO} front_nodes={len(front)} "
          f"total_nodes={len(nodes)} frames/tile={frames_per_tile} "
          f"out_frames={tiles * frames_per_tile}")
    print(f"saved {DST} {os.path.getsize(DST) / 1e6:.1f} MB")


def _rename(t, i, inits, scaled):
    if not t:
        return t
    if t in scaled:
        return f"{t}__t{i}"
    if t in inits:
        return t
    if t == "/Pad_output_0":        # the tile's own slice
        return f"tile{i}_in"
    return f"{t}__t{i}"


def _prune_dead(nodes, g):
    keep, live = [], {o.name for o in g.output}
    for n in reversed(nodes):
        if any(o in live for o in n.output):
            keep.append(n)
            live.update(i for i in n.input if i)
    keep.reverse()
    return keep


if __name__ == "__main__":
    main()
