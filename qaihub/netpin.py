#!/usr/bin/env python3
"""Keep the pinned edge IPs fresh.

SagerNet hands out fake-ip answers (198.18.x.x), so every host we talk to is
pinned to a real address.  CloudFront reassigns those addresses whenever the
WiFi reconnects, so resolve over DoH, probe each candidate with a real TLS
handshake, install an `ip rule ... lookup wlan0` for the survivors and cache the
result for the other scripts.

usage: netpin.py [host ...]        (defaults: the hub API + S3 data plane)
"""
import json
import os
import subprocess
import sys

import requests

DOH = "https://223.5.5.5/resolve"
CACHE = "/data/data/com.termux/files/usr/tmp/pins.json"
DEFAULT = ["workbench.aihub.qualcomm.com",
           "tetrahub-qprod-userdata.s3-accelerate.amazonaws.com"]


def resolve(host):
    r = requests.get(DOH, params={"name": host, "type": "A"}, timeout=10)
    return [a["data"] for a in r.json().get("Answer", []) if a["type"] == 1]


def alive(host, ip):
    try:
        requests.get(f"https://{host}/", timeout=8,
                     headers={"Host": host},
                     # certificate + SNI still carry the real host name
                     )
    except requests.exceptions.SSLError:
        return False
    except requests.exceptions.RequestException:
        return False
    return True


def alive_pinned(host, ip):
    """Probe one IP without touching the resolver: connect to ip, SNI = host."""
    import http.client
    import socket
    import ssl

    ctx = ssl.create_default_context()
    try:
        with socket.create_connection((ip, 443), timeout=8) as s:
            with ctx.wrap_socket(s, server_hostname=host):
                return True
    except Exception:                                   # noqa: BLE001
        return False


def add_rule(ip):
    subprocess.run(["sudo", "/system/bin/ip", "rule", "add", "pref", "50",
                    "to", ip, "lookup", "wlan0"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    hosts = sys.argv[1:] or DEFAULT
    pins = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for host in hosts:
        try:
            cands = resolve(host)
        except Exception as e:                          # noqa: BLE001
            print(f"{host}: DoH failed ({type(e).__name__})", flush=True)
            continue
        good = []
        for ip in cands:
            ok = alive_pinned(host, ip)
            print(f"  {host} {ip} {'ok' if ok else 'dead'}", flush=True)
            if ok:
                add_rule(ip)
                good.append(ip)
        if good:
            pins[host] = good
        print(f"{host}: {len(good)}/{len(cands)} alive", flush=True)
    json.dump(pins, open(CACHE, "w"), indent=1)
    print("saved", CACHE, flush=True)


if __name__ == "__main__":
    main()
