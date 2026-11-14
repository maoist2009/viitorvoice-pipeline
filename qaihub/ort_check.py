import numpy as np, onnxruntime as ort, os
W="/data/data/com.termux/files/usr/tmp/vv_nar"
M="/data/data/com.termux/files/home/projects/dreamlite/salvage/dl_vv/llm_backbone.onnx"
emb=np.load(os.path.join(W,"emb256.npy")).astype(np.float32).reshape(1,256,1024)
mask=np.zeros((1,1,256,256),dtype=np.float32)
s=ort.InferenceSession(M, providers=["CPUExecutionProvider"])
out=s.run(None,{"inputs_embeds":emb,"attention_mask":mask})[0]
a=out.reshape(256,1024).astype(np.float64)
b=np.load(os.path.join(W,"torch_hid.npy")).reshape(256,1024).astype(np.float64)
c=(a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
print("ORT(fp32 onnx) vs torch: cos mean %.6f min %.6f pos0 %.4f max|d| %.4f"%(
 c.mean(),c.min(),c[0],np.abs(a-b).max()))
g=np.load(os.path.join(W,"hid_fp16.npy")).reshape(256,1024).astype(np.float64)
c2=(g*a).sum(-1)/(np.linalg.norm(g,axis=-1)*np.linalg.norm(a,axis=-1))
print("NPU fp16 vs ORT: cos mean %.4f"%c2.mean())
