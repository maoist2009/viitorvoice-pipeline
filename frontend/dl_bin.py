#!/usr/bin/env python3
"""Parallel ranged download of a hub model to the phone."""
import json
import os
import queue
import sys
import threading
import time

import requests
import qai_hub as hub
from qai_hub import public_rest_api as p

model_id, out = sys.argv[1], sys.argv[2]
c = hub.Client()
url = p.get_model_download_url(c.config, model_id).url
parts = out + ".parts"
os.makedirs(parts, exist_ok=True)
r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
total = int(r.headers["Content-Range"].split("/")[1])
r.close()
ch = 8 << 20
nch = (total + ch - 1) // ch
print(f"download {total/1e6:.0f} MB in {nch} chunks", flush=True)
q = queue.Queue()
for i in range(nch):
    q.put(i)
lock = threading.Lock()
st = {"n": 0, "b": 0}
t0 = time.time()


def fetch(i):
    a, b = i * ch, min(i * ch + ch - 1, total - 1)
    want = b - a + 1
    f = f"{parts}/{i:05d}"
    if os.path.exists(f) and os.path.getsize(f) == want:
        return want
    for _ in range(8):
        try:
            rr = requests.get(url, headers={"Range": f"bytes={a}-{b}"},
                              timeout=(20, 120), stream=True)
            got = 0
            with open(f, "wb") as fh:
                for c2 in rr.iter_content(262144):
                    fh.write(c2)
                    got += len(c2)
            rr.close()
            if got == want:
                return want
        except Exception:
            pass
        time.sleep(1)
    return 0


def worker():
    while True:
        try:
            i = q.get_nowait()
        except queue.Empty:
            return
        n = fetch(i)
        with lock:
            st["n"] += 1
            st["b"] += n
            if st["n"] % 8 == 0 or st["n"] == nch:
                el = time.time() - t0
                print(f"{st['n']}/{nch} {st['b']/1e6:.0f} MB "
                      f"{st['b']/1e6/el:.2f} MB/s", flush=True)


ts = [threading.Thread(target=worker) for _ in range(12)]
[t.start() for t in ts]
[t.join() for t in ts]
missing = sum(1 for i in range(nch)
              if not (os.path.exists(f"{parts}/{i:05d}") and
                      os.path.getsize(f"{parts}/{i:05d}") ==
                      min((i + 1) * ch, total) - i * ch))
if missing:
    print(f"INCOMPLETE {missing} chunks missing", flush=True)
    sys.exit(1)
with open(out, "wb") as fo:
    for i in range(nch):
        with open(f"{parts}/{i:05d}", "rb") as fi:
            fo.write(fi.read())
print(f"DONE {out} {os.path.getsize(out)} in {time.time()-t0:.0f}s", flush=True)
