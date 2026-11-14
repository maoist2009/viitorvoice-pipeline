#!/usr/bin/env python3
"""Export the host-side embedding tables the NAR loop needs.

The compiled graphs start at `inputs_embeds`, so the gather that turns token ids
into vectors happens on the host:

    text position :  text_emb[id]                        (only row 0 of the 12)
    audio position:  sem_emb[id0] + sum_c aco_emb[id_c]

Everything is stored fp16 (the LLM graph consumes int16 activations, so fp16
tables cost nothing in accuracy) and row-major, one file per table.

usage: extract_embeddings.py <out_dir>
"""
import json
import os
import sys

import numpy as np
from safetensors import safe_open

CKPT = os.path.expanduser(
    "~/projects/voice_editor/sm8750_voice_editor/viitor-voice-nar/"
    "local_models/llm/0p6_emotion/model.safetensors")
TEXT_ROWS = 52697          # tokenizer max id + 1
CHUNK = 8192


def dump(f, key, rows, path):
    s = f.get_slice(key)
    out = np.empty((rows, 1024), dtype=np.float16)
    for r in range(0, rows, CHUNK):
        e = min(r + CHUNK, rows)
        out[r:e] = s[r:e].astype(np.float16)
    out.tofile(path)
    print(f"{key}: {rows}x1024 fp16 -> {path} {os.path.getsize(path)//1048576} MB", flush=True)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out, exist_ok=True)
    with safe_open(CKPT, framework="np") as f:
        dump(f, "llm.embed_tokens.weight", TEXT_ROWS, os.path.join(out, "text_emb.bin"))
        dump(f, "semantic_embedding.weight", 16387, os.path.join(out, "sem_emb.bin"))
        dump(f, "acoustic_embedding.weight", 11297, os.path.join(out, "aco_emb.bin"))
        offs = f.get_tensor("acoustic_codebook_offsets").astype(np.int32).tolist()
    meta = {"dim": 1024, "text_rows": TEXT_ROWS, "sem_rows": 16387, "aco_rows": 11297,
            "aco_offsets": offs,
            "mask_ids": [16386] + [1026] * 11,
            "pause_ids": [16384] + [1024] * 11,
            "sep_ids": [16385] + [1025] * 11,
            "sem_vocab": 16387, "aco_vocab": 1027, "num_codebooks": 12}
    with open(os.path.join(out, "emb_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
