#!/usr/bin/env python3
"""Download the model weights needed to re-export the lost .pt2 graphs.

Runner died on 2026-10-11; decoder/encoder/unet DLCs are already verified on
the phone, so only the sources for llm / w2v / clip re-export are pulled:
  ZzWater/ViiTorVoice-NAR  llm/0p6_emotion, dualcodec/w2v-bert-2.0
  carlofkl/DreamLite-base  text_encoder
"""
import os

from huggingface_hub import hf_hub_download

BASE = os.path.expanduser("~/dl_vv/models")

FILES = {
    "ZzWater/ViiTorVoice-NAR": [
        "llm/0p6_emotion/config.json",
        "llm/0p6_emotion/model.safetensors",
        "llm/0p6_emotion/tokenizer_config.json",
        "dualcodec/w2v-bert-2.0/config.json",
        "dualcodec/w2v-bert-2.0/preprocessor_config.json",
        "dualcodec/w2v-bert-2.0/model.safetensors",
        "config.json",
    ],
    "carlofkl/DreamLite-base": [
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
    ],
}


def main():
    for repo, rels in FILES.items():
        print(f"=== {repo}", flush=True)
        for rel in rels:
            dest_dir = os.path.join(BASE, repo.split("/")[-1], os.path.dirname(rel))
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.isfile(os.path.join(dest_dir, os.path.basename(rel))):
                print(f"  [have] {rel}", flush=True)
                continue
            print(f"  [get ] {rel}", flush=True)
            p = hf_hub_download(repo, rel, local_dir=os.path.join(
                BASE, repo.split("/")[-1]))
            print(f"        -> {os.path.getsize(p)/1e6:.1f} MB", flush=True)
    print("ALL_DOWNLOADS_OK", flush=True)


if __name__ == "__main__":
    main()
