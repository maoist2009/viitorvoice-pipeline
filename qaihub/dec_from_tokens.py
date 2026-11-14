#!/usr/bin/env python3
"""Decode an existing tokens.npy through the 5-graph decoder chain.

usage: dec_from_tokens.py <tokens.npy> <out.wav>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nar_proto as P  # noqa: E402


def main():
    toks = np.load(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(P.HERE, "nar_out.wav")
    P.decoder(toks, out)
    print("[done]", out, os.path.getsize(out), "B", flush=True)


if __name__ == "__main__":
    main()
