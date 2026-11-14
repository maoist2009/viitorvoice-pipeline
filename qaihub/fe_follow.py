#!/usr/bin/env python3
"""Follow a fe compile job to a local .bin: poll -> link -> poll -> download.

Compile jobs on the hub are cheap to re-poll; the API edge IP is rotated when
the pinned CloudFront IP times out, and the ~1 GB artifact is pulled in 8 MB
ranges over five HKG/SIN edges in parallel (measured ~15 MB/s aggregate).

usage: fe_follow.py <compile_job> <state.json> <out.bin>
"""
import itertools
import json
import os
import queue
import sys
import threading
import time

import requests
import urllib3

import pin_vv  # noqa: F401
import qai_hub as hub

urllib3.disable_warnings()

API_IPS = ["13.35.190.35", "13.35.190.127", "13.35.190.115"]
EDGES = ["13.35.221.233", "13.35.213.211", "13.226.64.144",
         "13.249.36.38", "13.226.71.165"]
HOST = "tetrahub-qprod-userdata.s3-accelerate.amazonaws.com"

COMPILE, STATE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DONE = ("SUCCESS", "FAILED", "CANCELLED", "ERROR")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def api(fn, *a, **kw):
    """Run an API call, rotating the pinned control-plane IP on failure."""
    err = None
    for ip in API_IPS:
        pin_vv.HOSTS["workbench.aihub.qualcomm.com"] = (ip,)
        pin_vv._cycle["workbench.aihub.qualcomm.com"] = itertools.cycle([ip])
        try:
            return fn(*a, **kw)
        except Exception as e:                       # noqa: BLE001
            err = e
    raise err


def state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(**kw):
    st = state()
    st.update(kw)
    json.dump(st, open(STATE, "w"))


def wait(job_id, label):
    while True:
        try:
            s = api(lambda: hub.get_job(job_id).get_status())
            if s.code in DONE:
                log(f"{label} {s.code} {s.message[:120]}")
                return s.code
        except Exception as e:                       # noqa: BLE001
            log(f"{label} poll error {type(e).__name__}")
        time.sleep(45)


def download(model_id):
    c = hub.Client()
    from qai_hub import public_rest_api as p
    u = api(lambda: p.get_model_download_url(c.config, model_id)).url
    pin_vv.HOSTS[HOST] = tuple(EDGES)
    pin_vv._cycle[HOST] = itertools.cycle(EDGES)

    parts = OUT + ".parts"
    os.makedirs(parts, exist_ok=True)
    r = requests.get(u, headers={"Range": "bytes=0-0"}, timeout=60)
    total = int(r.headers["Content-Range"].split("/")[1])
    r.close()
    CH = 8 << 20
    nch = (total + CH - 1) // CH
    log(f"download {total/1e6:.0f} MB in {nch} chunks")
    q = queue.Queue()
    for i in range(nch):
        q.put(i)
    lock = threading.Lock()
    st = {"done": 0, "bytes": 0}
    t0 = time.time()

    def fetch(i):
        a = i * CH
        b = min(a + CH - 1, total - 1)
        want = b - a + 1
        f = f"{parts}/{i:05d}"
        if os.path.exists(f) and os.path.getsize(f) == want:
            return want
        for _ in range(8):
            try:
                rr = requests.get(u, headers={"Range": f"bytes={a}-{b}"},
                                  timeout=(20, 60), stream=True)
                got = 0
                with open(f, "wb") as fh:
                    for c2 in rr.iter_content(262144):
                        fh.write(c2)
                        got += len(c2)
                rr.close()
                if got == want:
                    return want
            except Exception:                        # noqa: BLE001
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
                st["done"] += 1
                st["bytes"] += n
                if st["done"] % 10 == 0:
                    el = time.time() - t0
                    log(f"{st['done']}/{nch} {st['bytes']/1e6:.0f} MB "
                        f"{st['bytes']/1e6/el:.2f} MB/s")

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if st["done"] != nch:
        log(f"INCOMPLETE {st['done']}/{nch}")
        return False
    with open(OUT, "wb") as f:
        for i in range(nch):
            with open(f"{parts}/{i:05d}", "rb") as g:
                f.write(g.read())
    log(f"bin {OUT} {os.path.getsize(OUT)/1e6:.0f} MB")
    return True


def main():
    st = state()
    if "link" not in st:
        code = wait(COMPILE, "compile")
        if code != "SUCCESS":
            return 1
        j = api(lambda: hub.get_job(COMPILE))
        lj = api(lambda: hub.submit_link_job(models=[j.get_target_model()],
                                            device=j.device,
                                            name=COMPILE + "_link",
                                            options="--qairt_version 2.51"))
        save(link=lj.job_id)
        log(f"link job {lj.job_id}")
        st = state()
    code = wait(st["link"], "link")
    if code != "SUCCESS":
        return 2
    j = api(lambda: hub.get_job(st["link"]))
    m = j.get_target_model()
    log(f"target model {m.model_id}")
    return 0 if download(m.model_id) else 3


if __name__ == "__main__":
    sys.exit(main())
