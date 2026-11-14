import itertools
import os
import queue
import shutil
import sys
import threading
import time

import requests
import urllib3

urllib3.disable_warnings()
sys.path.insert(0, "/data/data/com.termux/files/home/projects/rvc/rvc-android/salvage/dl_vv")
import pin_vv  # noqa: E402

URL = open("/data/data/com.termux/files/usr/tmp/dlurl.txt").read().strip()
HOST = "tetrahub-qprod-userdata.s3-accelerate.amazonaws.com"
IPS = ["13.35.221.233", "13.35.213.211", "13.226.64.144",
       "13.249.36.38", "13.226.71.165"]
OUT = "/data/data/com.termux/files/usr/tmp/seg_fea16.bin"
PARTS = "/data/data/com.termux/files/usr/tmp/fea16_parts"
CH = 8 * 1024 * 1024
NTHREADS = 12

shutil.rmtree(PARTS, ignore_errors=True)
os.makedirs(PARTS, exist_ok=True)

r = requests.get(URL, headers={"Range": "bytes=0-0"}, timeout=60)
total = int(r.headers["Content-Range"].split("/")[1])
r.close()
nch = (total + CH - 1) // CH
print("total", total, "chunks", nch, "threads", NTHREADS, flush=True)

# round-robin the edges for every new connection
pin_vv.HOSTS[HOST] = tuple(IPS)
pin_vv._cycle[HOST] = itertools.cycle(IPS)

lock = threading.Lock()
state = {"done": 0, "bytes": 0}
q = queue.Queue()
for i in range(nch):
    q.put(i)


def fetch(i):
    a = i * CH
    b = min(a + CH - 1, total - 1)
    want = b - a + 1
    p = f"{PARTS}/{i:05d}"
    if os.path.exists(p) and os.path.getsize(p) == want:
        return want
    for _ in range(8):
        try:
            rr = requests.get(URL, headers={"Range": f"bytes={a}-{b}"},
                              timeout=(20, 60), stream=True)
            buf = 0
            with open(p, "wb") as f:
                for c in rr.iter_content(262144):
                    f.write(c)
                    buf += len(c)
            rr.close()
            if buf == want:
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
            state["done"] += 1
            state["bytes"] += n
            if state["done"] % 10 == 0:
                el = time.time() - t0
                print(f"{state['done']}/{nch} {state['bytes']/1e6:.0f} MB "
                      f"{state['bytes']/1e6/el:.2f} MB/s", flush=True)


t0 = time.time()
ts = [threading.Thread(target=worker, daemon=True) for _ in range(NTHREADS)]
[t.start() for t in ts]
[t.join() for t in ts]
el = time.time() - t0
print("elapsed %.0fs avg %.2f MB/s got %.0f MB" %
      (el, state["bytes"] / 1e6 / el, state["bytes"] / 1e6), flush=True)

if state["done"] == nch:
    with open(OUT, "wb") as f:
        for i in range(nch):
            with open(f"{PARTS}/{i:05d}", "rb") as g:
                f.write(g.read())
    print("out", os.path.getsize(OUT), flush=True)
else:
    print("INCOMPLETE", state["done"], nch, flush=True)
