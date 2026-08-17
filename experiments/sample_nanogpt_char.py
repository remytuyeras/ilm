"""Generate a deterministic character-level completion from nanoGPT."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, ROOT_DIR)

from nanogpt_char_utils import get_device, load_character_data, load_nanogpt_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a frozen nanoGPT character checkpoint.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--nanogpt-dir", default="baselines/nanoGPT")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-characters", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--generation-seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-file", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_new_characters <= 0:
        raise SystemExit("--max-new-characters must be positive")
    if args.temperature < 0:
        raise SystemExit("--temperature must be zero or positive")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    random.seed(args.generation_seed)
    np.random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.generation_seed)

    device = get_device(args.device)
    meta = load_character_data(args.data_dir)
    stoi, itos = meta["stoi"], meta["itos"]
    unit = meta.get("unit", "character")
    prompt_units = args.prompt if unit == "character" else args.prompt.encode("utf-8")
    missing = sorted(set(prompt_units) - set(stoi))
    if missing:
        raise SystemExit(f"prompt contains units outside the frozen vocabulary: {missing!r}")
    model, _ = load_nanogpt_checkpoint(args.checkpoint_path, args.nanogpt_dir, device)
    context = torch.tensor([[stoi[value] for value in prompt_units]], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=args.max_new_characters,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    generated_units = [itos[int(token)] for token in generated[0].tolist()]
    if unit == "character":
        text = "".join(generated_units)
        completion = text[len(args.prompt):]
    elif unit == "utf8-byte":
        completion = bytes(generated_units[len(prompt_units):]).decode("utf-8", errors="replace")
    else:
        raise SystemExit(f"unsupported nanoGPT data unit {unit!r}")
    print(completion)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({
                "checkpoint_path": args.checkpoint_path,
                "prompt": args.prompt,
                "completion": completion,
                "generation_seed": args.generation_seed,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "max_new_characters": args.max_new_characters,
            }, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
