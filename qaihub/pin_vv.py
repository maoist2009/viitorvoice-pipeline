"""Pin qai-hub's control + S3 data plane to reachable real IPs.

Same recipe as tools/asr/s3pin.py (verified 2026-11-10): SagerNet's TUN
hands out fake-ip answers (198.18.x.x), so the hostnames must be resolved
here and the kernel sent out wlan0 with

    ip rule add pref 50 to 3.163.84.103 lookup wlan0
    ip rule add pref 50 to 13.35.190.0/24 lookup wlan0

(netd wipes those rules on a WiFi reconnect; re-add them if transfers
suddenly start timing out again.)
"""
from __future__ import annotations

import itertools
import os
import socket

_EDGES = {
    # 2026-11-10 实测：TLS 642 ms，直连 403（可达）
    "best": ("3.163.84.103",),
    "nrt": ("13.33.11.133", "3.175.255.225"),
    "us": ("18.238.104.192", "3.163.84.103", "18.155.208.44"),
}
_EDGES["all"] = _EDGES["nrt"] + _EDGES["us"]

_env = os.environ.get("S3PIN_IP")
EDGE = (_env,) if _env else _EDGES.get(os.environ.get("S3PIN_EDGE", "best"), _EDGES["best"])

HOSTS = {
    "tetrahub-qprod-userdata.s3-accelerate.amazonaws.com": EDGE,
    "workbench.aihub.qualcomm.com":
        (os.environ.get("API_IP") or "13.35.190.35",),
}

_orig = socket.getaddrinfo
_cycle = {h: itertools.cycle(ips) for h, ips in HOSTS.items()}


def _patched(host, port, family=0, type=0, proto=0, flags=0):
    ips = _cycle.get(host)
    if ips is None:
        return _orig(host, port, family, type, proto, flags)
    ip = next(ips)
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
             (ip, port if isinstance(port, int) else int(port)))]


def install():
    socket.getaddrinfo = _patched


install()
