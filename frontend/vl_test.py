import time, wave, gc
import numpy as np, onnxruntime as ort
from scipy.signal import resample_poly
SR, CORE, FE_S, FR = 24000, 192000, 128640, 200
w = wave.open("ref.wav"); n = w.getnframes()
pcm = np.frombuffer(w.readframes(n), np.int16).astype(np.float32)/32768.0
print("ref samples", n)
a16f = resample_poly(pcm, 2, 3).astype(np.float32)
n16 = min(len(a16f), FE_S)
a16 = np.zeros(FE_S, np.float32); a16[:n16] = a16f[:n16]
print("n16", n16)
so = ort.SessionOptions(); so.intra_op_num_threads = 2
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.enable_cpu_mem_arena = False
t0=time.time(); fe = ort.InferenceSession("w2v_front8s.onnx", so, providers=["CPUExecutionProvider"]); print("load %.1f s"%(time.time()-t0))
outs={}
for tag, v in (("n16", n16), ("full", FE_S)):
    t0=time.time(); o = fe.run(None, {"audio16k": a16[None,:], "valid_len": np.array([v], np.float32)})[0]
    print("%s valid=%d  %.2f s  sum=%.4f"%(tag, v, time.time()-t0, float(o.sum())))
    outs[tag]=o.copy()
f = min(n//960, FR)
a, b = outs["n16"], outs["full"]
d = np.abs(a-b)[:, :, :f]; ref = np.abs(a[:, :, :f])
print("first %d frames: maxabs %.5f  relL2 %.6f"%(f, d.max(), np.linalg.norm(d)/np.linalg.norm(ref)))
dc = d.reshape(1024, -1).max(axis=0)
print("per-frame maxdiff: frame0 %.5f  frame%d %.5f  frame%d %.5f"%(dc[0], f//2, dc[f//2], f-1, dc[f-1]))
too = dc > 0.05
print("frames with diff>0.05:", int(too.sum()), "of", f)
np.save("feat_n16.npy", outs["n16"]); np.save("feat_full.npy", outs["full"])
