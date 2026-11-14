#!/usr/bin/env python3
"""Recompile llm / encoder with fp32 IO.

Both were first compiled with `--quantize_io`, which makes qai-hub pick a fixed
heuristic IO range ([0, 1] for every input, scale 1/255, offset 0). Their inputs
are signed (LLM input embeds are ~N(0,1); the codec encoder takes a [-1,1]
waveform and w2v hidden states), so every negative value was clipped to 0 and
the compiled graphs are numerically useless.

Dropping `--quantize_io` keeps the same int8 internals but exposes float32 IO,
exactly like the w2v graph (which was compiled without it and is correct).

Nothing is re-uploaded: each source CompileJob still carries its model and its
calibration dataset.

usage: fixio_jobs.py submit | poll | link | poll-link | fetch
"""
import json
import os
import sys

import qai_hub as hub

OPTS = ("--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type int8 "
        "--qairt_version 2.51")
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixio_jobs.json")
SRC = {
    "llm": "j57o7d8vg",
    "encoder": "jgd6872lp",
}


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(d):
    json.dump(d, open(STATE, "w"), indent=1)


def main():
    cmd = sys.argv[1]
    c = hub.Client()
    st = load()

    if cmd == "submit":
        for name, jid in SRC.items():
            if name in st.get("compile", {}):
                print(name, "already", st["compile"][name])
                continue
            src = c.get_job(jid)
            j = c.submit_compile_job(model=src.model, device=src.device,
                                     name=f"{name}_fp32io",
                                     options=OPTS,
                                     calibration_data=src.calibration_dataset)
            st.setdefault("compile", {})[name] = j.job_id
            print(name, "COMPILE", j.job_id, flush=True)
        save(st)

    elif cmd == "poll":
        for name, jid in st.get("compile", {}).items():
            s = c.get_job(jid).get_status()
            print(f"{name:<8} {s.code:<18} {s.message[:120]}", flush=True)

    elif cmd == "link":
        st.setdefault("link", {})
        for name, jid in st.get("compile", {}).items():
            j = c.get_job(jid)
            if j.get_status().code != "SUCCESS" or name in st["link"]:
                continue
            lj = c.submit_link_job(models=j.get_target_model(), device=j.device,
                                   name=f"{name}_fp32io_link",
                                   options="--qairt_version 2.51")
            st["link"][name] = lj.job_id
            print(name, "LINK", lj.job_id, flush=True)
        save(st)

    elif cmd == "poll-link":
        for name, jid in st.get("link", {}).items():
            s = c.get_job(jid).get_status()
            print(f"{name:<8} {s.code:<18} {s.message[:120]}", flush=True)

    elif cmd == "fetch":
        out = os.path.join(os.path.dirname(STATE), "fixio")
        os.makedirs(out, exist_ok=True)
        for name, jid in st.get("link", {}).items():
            j = c.get_job(jid)
            if j.get_status().code != "SUCCESS":
                print(name, "not ready:", j.get_status().code)
                continue
            dest = os.path.join(out, f"{name}.bin")
            j.download_target_model(dest)
            print(name, "->", dest, os.path.getsize(dest), "B", flush=True)

    else:
        raise SystemExit("unknown command")


if __name__ == "__main__":
    main()
