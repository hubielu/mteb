"""Reproduce the MoCa-Qwen25VL-3B model-card example through the mteb wrapper.

Run this before any benchmark job: it costs ~2 minutes of GPU time and catches the
two failure modes that would otherwise silently produce garbage numbers, namely
(1) checkpoint keys that do not map onto the installed transformers version and
(2) an attention implementation that keeps the backbone causal.

Usage:
    python check_moca_parity.py --image figures/example.jpg
    python check_moca_parity.py --image figures/example.jpg --attn sdpa
"""

from __future__ import annotations

import argparse

import torch
import transformers
from PIL import Image

from mteb.models.model_implementations.moca_models import (
    _IMAGE_TOKEN,
    MoCaWrapper,
)

# Expected values, copied from https://huggingface.co/moca-embed/MoCa-Qwen25VL-3B
EXPECTED = {
    "image+text -> 'A cat and a dog'": 0.4824,
    "image+text -> 'A cat and a tiger'": 0.3613,
    "text 'A cat and a dog' -> image": 0.3945,
    "text 'A cat and a tiger' -> image": 0.3184,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moca-embed/MoCa-Qwen25VL-3B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--processor", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", default="figures/example.jpg")
    parser.add_argument("--attn", default=None, help="flash_attention_2 | sdpa | eager")
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    print(f"transformers {transformers.__version__} | torch {torch.__version__}")

    model = MoCaWrapper(
        args.model,
        revision=args.revision,
        processor_name=args.processor,
        attn_implementation=args.attn,
    )
    print(f"attn implementation: {model.model.config._attn_implementation}")
    print(
        f"is_causal on layer 0: {model.model.language_model.layers[0].self_attn.is_causal}"
    )

    image = Image.open(args.image)

    # Image + text query against two text targets
    query = model._embed(
        [
            _IMAGE_TOKEN
            + "Represent the given image with the following question: What is in the image\n"
        ],
        [image],
    )
    cat_dog = model._embed(["A cat and a dog"])
    cat_tiger = model._embed(["A cat and a tiger"])

    # Text query against the image as target
    doc = model._embed([_IMAGE_TOKEN + "Represent the given image.\n"], [image])
    q_dog = model._embed(
        ["Find me an everyday image that matches the given caption: A cat and a dog.\n"]
    )
    q_tiger = model._embed(
        [
            "Find me an everyday image that matches the given caption: A cat and a tiger.\n"
        ]
    )

    got = {
        "image+text -> 'A cat and a dog'": (query @ cat_dog.T).item(),
        "image+text -> 'A cat and a tiger'": (query @ cat_tiger.T).item(),
        "text 'A cat and a dog' -> image": (q_dog @ doc.T).item(),
        "text 'A cat and a tiger' -> image": (q_tiger @ doc.T).item(),
    }

    print(f"\n{'case':<40} {'expected':>9} {'got':>9} {'delta':>8}  ok")
    all_ok = True
    for name, expected in EXPECTED.items():
        value = got[name]
        delta = value - expected
        ok = abs(delta) <= args.tolerance
        all_ok &= ok
        print(f"{name:<40} {expected:>9.4f} {value:>9.4f} {delta:>+8.4f}  {ok}")

    print(f"\nPARITY {'PASS' if all_ok else 'FAIL'} (tolerance {args.tolerance})")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
