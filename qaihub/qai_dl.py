#!/usr/bin/env python3
"""qai_dl.py — pull a qai-hub compile job's target model straight from the
qai-hub S3 data plane, bypassing the (ephemeral, bandwidth-capped) runner.

Range-based and resumable: appends to <dest>.part, slices refreshed against
the live object size, STS credentials re-fetched before they expire, md5
verified at the end.

usage:  qai_dl.py <job_id> <dest> <expected_md5>
"""
import hashlib
import os
import socket
import sys
import time

import boto3
import qai_hub
from qai_hub.client import api

SLICE = 32 << 20  # 32 MB

# Optional: pin the accelerate edge IPs if DNS/TLS misbehaves on this link.
# Comma-separated list in env QAI_S3_PINS.
EDGE_PINS = os.environ.get("QAI_S3_PINS", "")


def maybe_pin():
    if not EDGE_PINS:
        return
    pins = [p.strip() for p in EDGE_PINS.split(",") if p.strip()]
    real = socket.getaddrinfo

    def patched(host, *a, **kw):
        if "s3-accelerate" in host and pins:
            out = []
            for p in pins:
                out.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (p, 443)))
            return out
        return real(host, *a, **kw)

    socket.getaddrinfo = patched


def fetch_creds(model_id):
    j = qai_hub.get_job(job_id_global)
    tm = j.get_target_model()
    cfg = tm._owner.config
    resp = api._get_model_download_response(cfg, model_id, True)
    if not resp.HasField("credentials"):
        raise RuntimeError("download response carried no STS credentials")
    return resp.credentials


def make_client(c):
    # Mirror qai_hub.s3_transfer._build_client exactly: an explicit
    # endpoint_url breaks SigV4 on the accelerate host (403 on every op).
    import botocore.session
    from botocore.client import Config

    session = botocore.session.get_session()
    return session.create_client(
        "s3",
        region_name=c.region or None,
        aws_access_key_id=c.access_key_id,
        aws_secret_access_key=c.secret_access_key,
        aws_session_token=c.session_token,
        config=Config(
            signature_version="s3v4",
            s3={"use_accelerate_endpoint": bool(c.use_transfer_acceleration)},
        ),
    )


def object_size(cl, bucket, key):
    return cl.head_object(Bucket=bucket, Key=key)["ContentLength"]


def main():
    global job_id_global
    job_id_global = sys.argv[1]
    dest = sys.argv[2]
    want_md5 = sys.argv[3].lower() if len(sys.argv) > 3 else None
    part = dest + ".part"
    maybe_pin()

    j = qai_hub.get_job(job_id_global)
    tm = j.get_target_model()
    mid = tm.model_id
    print(f"[{os.path.basename(dest)}] job {job_id_global} model {mid}", flush=True)

    c = fetch_creds(mid)
    print(f"[{os.path.basename(dest)}] bucket {c.bucket} key {c.key} accel={bool(c.use_transfer_acceleration)}", flush=True)
    cl = make_client(c)
    total = object_size(cl, c.bucket, c.key)
    print(f"[{os.path.basename(dest)}] object size {total} bytes ({total >> 20} MB)", flush=True)

    have = os.path.getsize(part) if os.path.exists(part) else 0
    h = hashlib.md5()
    if have:
        # seed the md5 with what we already have
        with open(part, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        print(f"[{os.path.basename(dest)}] resuming from {have} bytes", flush=True)

    t0 = last = time.time()
    last_bytes = have
    while have < total:
        end = min(have + SLICE, total) - 1
        try:
            r = cl.get_object(Bucket=c.bucket, Key=c.key, Range=f"bytes={have}-{end}")
            chunk = r["Body"].read()
        except Exception as e:
            print(f"[{os.path.basename(dest)}] slice {have} failed: {e}; refreshing creds", flush=True)
            c = fetch_creds(mid)
            cl = make_client(c)
            time.sleep(2)
            continue
        if not chunk:
            raise RuntimeError("empty slice")
        with open(part, "ab") as f:
            f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        h.update(chunk)
        have += len(chunk)
        now = time.time()
        if now - last >= 20:
            rate = (have - last_bytes) / (now - last) / (1 << 20)
            eta = (total - have) / max(1e-9, (have - last_bytes) / (now - last))
            print(f"[{os.path.basename(dest)}] {have >> 20}/{total >> 20} MB "
                  f"({100 * have / total:.1f}%) {rate:.2f} MB/s eta {eta / 60:.0f}m", flush=True)
            last, last_bytes = now, have

    got = h.hexdigest()
    ok = (not want_md5) or got == want_md5
    if ok:
        os.rename(part, dest)
        print(f"[{os.path.basename(dest)}] OK ({total >> 20} MB) md5 {got}", flush=True)
    else:
        print(f"[{os.path.basename(dest)}] md5 MISMATCH got {got} want {want_md5} "
              f"(kept {part})", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
