#!/usr/bin/env python3
"""Peak RSS per part while it executes (what the app has to budget).

Each qnn-net-run run is polled from /proc/<pid>/status VmHWM; VmHWM is the
process high-water mark, so sampling once right before exit is enough.
"""
import os
import subprocess
import sys
import time

QNN = os.path.expanduser("~/qairt/2.51.40")
CL, BN, SK = (f"{QNN}/lib/aarch64-android", f"{QNN}/bin/aarch64-android",
              f"{QNN}/lib/hexagon-v79/unsigned")
bins, work = sys.argv[1], sys.argv[2]
env = dict(os.environ, LD_LIBRARY_PATH=f"/vendor/lib64:{CL}", ADSP_LIBRARY_PATH=SK)


def peak(bin_name, paths, label):
    lst = os.path.join(work, f"m_{bin_name}.txt")
    with open(lst, "w") as f:
        f.write(" ".join(paths) + "\n")
    out = os.path.join(work, f"m_{bin_name}")
    cmd = [f"{BN}/qnn-net-run", "--backend", f"{CL}/libQnnHtp.so",
           "--retrieve_context", os.path.join(bins, bin_name),
           "--input_list", lst, "--output_dir", out, "--use_native_input_files"]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hwm = 0
    while p.poll() is None:
        try:
            with open(f"/proc/{p.pid}/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        hwm = max(hwm, int(line.split()[1]))
                        break
        except OSError:
            pass
        time.sleep(0.02)
    print(f"{label:<12} peak RSS {hwm / 1024:7.1f} MB   (exit {p.returncode})", flush=True)
    return hwm


p1 = peak("p1.bin", [os.path.join(work, "sem.raw"), os.path.join(work, "aco.raw")], "p1 (fp16)")
p2 = peak("p2.bin", [os.path.join(work, "conv.raw")], "p2")
p3 = peak("p3.bin", [os.path.join(work, "c1.raw")], "p3")
p4 = peak("p4.bin", [os.path.join(work, "c2.raw")], "p4")
p5 = peak("p5chunk48.bin", [os.path.join(work, "c3_0.raw")], "p5 (1 chunk)")
print(f"\nlargest single context: {max(p1, p2, p3, p4, p5) / 1024:.0f} MB")
print(f"sum of weights (bins): {sum(os.path.getsize(os.path.join(bins, b)) for b in ('p1.bin','p2.bin','p3.bin','p4.bin','p5chunk48.bin')) / 1048576:.0f} MB")
