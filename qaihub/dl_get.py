#!/usr/bin/env python3
"""Download DLCs from the runner with md5 verification and resume-on-stall.

The cloudflared tunnel silently drops long-lived SFTP connections, and
paramiko's read timeout does not fire through the half-dead socketpair, so a
whole-file sftp.get() hangs forever partway. Fix: one fresh SSH connection per
SLICE bytes -- connect, seek, read at most SLICE, close, repeat. Short
connections survive (each runner.py invocation works), long ones do not.
"""
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner as R

REMOTE_DIR = "/home/runner/dl_vv/dlc"
LOCAL_DIR = os.path.expanduser("~/tmp/dl_vv/dlc")
FILES = sys.argv[1:] or ["decoder.dlc", "encoder.dlc", "unet.dlc", "llm.dlc"]
CHUNK = 1 << 22
SLICE = 64 << 20
READ_TIMEOUT = 90


def local_md5(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def new_link(tries=20, delay=4, deadline=45):
    """R.connect() can hang forever: its socketpair has no timeout, so when a
    spawned cloudflared never brings the tunnel up, paramiko's banner read
    blocks indefinitely. Race the connect against a watchdog thread that
    terminates the stuck cloudflared, then dial again."""
    import threading
    for attempt in range(tries):
        holder = {}
        done = threading.Event()

        def go():
            try:
                link = Link.__new__(Link)
                link.cli, link.proc = R.connect()
                holder["link"] = link
            except Exception as e:
                holder["err"] = e
            finally:
                done.set()

        threading.Thread(target=go, daemon=True).start()
        if done.wait(deadline):
            if "link" in holder:
                return holder["link"]
            err = holder.get("err")
            if err is not None:
                print(f"[connect] attempt {attempt}: {type(err).__name__}: "
                      f"{str(err)[:80]}", flush=True)
                time.sleep(delay)
                continue
        # timed out: the watchdog thread is still blocked in R.connect(); kill
        # its cloudflared so the socketpair unblocks and the thread can exit
        print(f"[connect] attempt {attempt}: no banner in {deadline}s, "
              f"killing cloudflared", flush=True)
        _kill_stuck_cloudflared()
        time.sleep(delay)
    raise RuntimeError("could not connect after %d tries" % tries)


def _kill_stuck_cloudflared():
    import subprocess
    try:
        subprocess.run(
            ["pkill", "-f", "access tcp --hostname rapidly-tall"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        pass


class Link:
    """One SSH connection. Construct via new_link() -- a bare Link() has no
    cli/proc because the tunnel connect can fail at any moment."""

    def close(self):
        try:
            self.cli.close()
        finally:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def run(self, cmd, timeout=180):
        chan = self.cli.get_transport().open_session()
        chan.settimeout(timeout)
        chan.exec_command(cmd)
        out = b""
        while True:
            if chan.recv_ready():
                out += chan.recv(65536)
            if chan.recv_stderr_ready():
                out += chan.recv_stderr(65536)
            if chan.exit_status_ready() and not chan.recv_ready():
                break
        return chan.recv_exit_status(), out.decode(errors="replace")

    def remote_meta(self, path):
        rc, out = self.run(f'stat -c "%s" {path} 2>/dev/null && md5sum {path}')
        if rc != 0 or not out.strip():
            return None, None
        lines = out.strip().splitlines()
        size = int(lines[0].split()[0])
        md5 = lines[-1].split()[0]
        return size, md5

    def pull_slice(self, remote, local_part, have, size):
        """Append remote[have : min(have+SLICE, size)] and close the handle.
        Returns True when the whole file has been transferred."""
        end = min(have + SLICE, size)
        sftp = self.cli.open_sftp()
        sftp.get_channel().settimeout(READ_TIMEOUT)
        try:
            rf = sftp.open(remote, "rb")
            try:
                rf.seek(have)
            except Exception:
                rf.close()
                rf = sftp.open(remote, "rb")
                rf.prefetch(size)
                rf.seek(have)
            with open(local_part, "ab") as lf:
                total = have
                while total < end:
                    n = min(CHUNK, end - total)
                    data = rf.read(n)
                    if not data:
                        raise EOFError("remote read returned empty before EOF")
                    lf.write(data)
                    total += len(data)
                lf.flush()
                os.fsync(lf.fileno())
        finally:
            try:
                sftp.close()
            except Exception:
                pass
        return end >= size


def fetch(name, max_slices=400):
    remote = os.path.join(REMOTE_DIR, name)
    local = os.path.join(LOCAL_DIR, name)
    part = local + ".part"
    os.makedirs(LOCAL_DIR, exist_ok=True)

    link = new_link()
    size, want = link.remote_meta(remote)
    if size is None:
        print(f"[{name}] MISSING on runner", flush=True)
        link.close()
        return False
    if os.path.isfile(local) and local_md5(local) == want:
        print(f"[{name}] already current ({want[:8]})", flush=True)
        link.close()
        return True
    link.close()

    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if have > size:
        print(f"[{name}] stale part {have} > {size}, restarting", flush=True)
        os.remove(part)
        have = 0

    t0 = time.time()
    slices = 0
    while have < size:
        if slices >= max_slices:
            print(f"[{name}] GAVE UP after {slices} slices", flush=True)
            return False
        try:
            link = new_link()
        except Exception as e:
            # the tunnel hiccups between slices; a failed connect must not kill
            # the whole fetch -- back off and dial again
            print(f"[{name}] connect failed: {type(e).__name__}: "
                  f"{str(e)[:90]} -> backoff", flush=True)
            time.sleep(5)
            continue
        try:
            done = link.pull_slice(remote, part, have, size)
            have = os.path.getsize(part)
            slices += 1
            if done:
                break
            el = time.time() - t0
            print(f"[{name}] {have/1e6:6.0f}/{size/1e6:.0f} MB "
                  f"({have/size*100:4.1f}%) {el:5.0f}s "
                  f"{have/el/1e6:.2f} MB/s", flush=True)
        except Exception as e:
            print(f"[{name}] slice @{have}: {type(e).__name__}: "
                  f"{str(e)[:90]} -> reconnect", flush=True)
            have = os.path.getsize(part) if os.path.isfile(part) else 0
        finally:
            link.close()

    got = local_md5(part)
    dt = time.time() - t0
    print(f"[{name}] done {dt:.0f}s md5 {got and got[:8]} want {want[:8]}",
          flush=True)
    if got != want:
        print(f"[{name}] md5 MISMATCH", flush=True)
        return False
    os.replace(part, local)
    print(f"[{name}] OK ({os.path.getsize(local)/1e6:.1f} MB)", flush=True)
    return True


def main():
    ok = True
    for name in FILES:
        ok = fetch(name) and ok
    print("ALL_OK" if ok else "SOME_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
