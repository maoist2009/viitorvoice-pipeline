#!/usr/bin/env python3
"""Runner control via cloudflared tunnel + paramiko (per feedback_ssh_use_python_lib).

Usage:
  LD_PRELOAD=... python3 runner.py <command...>   # run command on runner
  LD_PRELOAD=... python3 runner.py --put <local> <remote>
  LD_PRELOAD=... python3 runner.py --get <remote> <local>
"""
import os
import socket
import subprocess
import sys
import threading

import paramiko

HOST = "rapidly-tall-resist-continued.trycloudflare.com"
USER = "runner"
PASSWORD = "123456"
PORT = 22

LD_PRELOAD = "/data/data/com.termux/files/usr/lib/libpython3.14.so"


def open_tunnel_sock():
    """cloudflared access tcp bridges stdin/stdout to the origin SSH server in
    the foreground -- there is no local listening port. Expose that stream as a
    genuine socket via a socketpair plus two pump threads, so paramiko gets a
    real socket (recv/send/settimeout/close) to run SSH over."""
    proc = subprocess.Popen(
        ["cloudflared", "access", "tcp", "--hostname", HOST],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    local, remote = socket.socketpair()
    local.settimeout(None)
    remote.settimeout(None)

    def pump_up():  # cloudflared stdout -> paramiko socket
        try:
            while True:
                data = os.read(proc.stdout.fileno(), 65536)
                if not data:
                    break
                local.sendall(data)
        except OSError:
            pass
        finally:
            try:
                local.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def pump_down():  # paramiko socket -> cloudflared stdin
        try:
            while True:
                data = local.recv(65536)
                if not data:
                    break
                proc.stdin.write(data)
                proc.stdin.flush()
        except OSError:
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    for target, name in ((pump_up, "cf-up"), (pump_down, "cf-down")):
        threading.Thread(target=target, name=name, daemon=True).start()

    def drain_stderr():
        try:
            for line in proc.stderr:
                sys.stderr.write(f"[cf] {line.decode(errors='replace')}")
                sys.stderr.flush()
        except OSError:
            pass

    threading.Thread(target=drain_stderr, name="cf-err", daemon=True).start()
    return remote, proc


def connect():
    sock, proc = open_tunnel_sock()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        hostname="localhost", port=PORT, username=USER, password=PASSWORD,
        sock=sock, timeout=60, allow_agent=False, look_for_keys=False,
    )
    return cli, proc


def run_remote(cli, cmd, timeout=3600):
    chan = cli.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out = b""
    while True:
        if chan.recv_ready():
            out += chan.recv(65536)
        if chan.recv_stderr_ready():
            sys.stderr.buffer.write(chan.recv_stderr(65536))
            sys.stderr.buffer.flush()
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
    rc = chan.recv_exit_status()
    sys.stdout.write(out.decode(errors="replace"))
    sys.stdout.flush()
    return rc


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    cli, proc = connect()
    try:
        if args[0] == "--put":
            sftp = cli.open_sftp()
            sftp.put(args[1], args[2])
            sftp.close()
            print(f"[put] {args[1]} -> {args[2]}")
        elif args[0] == "--get":
            sftp = cli.open_sftp()
            sftp.get(args[1], args[2])
            sftp.close()
            print(f"[get] {args[1]} <- {args[2]}")
        else:
            rc = run_remote(cli, " ".join(args))
            sys.exit(rc)
    finally:
        cli.close()
        proc.terminate()


if __name__ == "__main__":
    main()
