"""Compute exact teacher-forced test BPB for a nanoGPT character checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, ROOT_DIR)

from nanogpt_char_utils import get_device, load_character_data, load_nanogpt_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute exact next-character likelihood in bits per UTF-8 byte."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--nanogpt-dir", default="baselines/nanoGPT")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--test-text", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument(
        "--evaluation-mode",
        choices=["full-context", "block-reset"],
        default="full-context",
        help=(
            "Score each event with its full available context, or score contiguous "
            "non-overlapping context blocks as used during training."
        ),
    )
    parser.add_argument("--output-file", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.evaluation_batch_size <= 0:
        raise SystemExit("--evaluation-batch-size must be positive")

    device = get_device(args.device)
    meta = load_character_data(args.data_dir)
    unit = meta.get("unit", "character")
    ids = np.memmap(Path(args.data_dir) / "test.bin", dtype=np.uint16, mode="r")
    if len(ids) < 2:
        raise SystemExit("test.bin must contain at least two character IDs")
    raw_test_bytes = Path(args.test_text).read_bytes()
    if unit == "character":
        raw_test_text = raw_test_bytes.decode("utf-8")
        if len(raw_test_text) != len(ids):
            raise SystemExit("test text and test.bin have different character counts")
        byte_weights = torch.tensor(
            [len(character.encode("utf-8")) for character in raw_test_text],
            dtype=torch.float64,
        )
        event_name = "character"
    elif unit == "utf8-byte":
        if len(raw_test_bytes) != len(ids):
            raise SystemExit("test bytes and test.bin have different event counts")
        byte_weights = torch.ones(len(ids), dtype=torch.float64)
        event_name = "byte"
    else:
        raise SystemExit(f"unsupported nanoGPT data unit {unit!r}")

    model, checkpoint = load_nanogpt_checkpoint(args.checkpoint_path, args.nanogpt_dir, device)
    block_size = int(checkpoint["model_args"]["block_size"])
    ids_tensor = torch.from_numpy(np.asarray(ids, dtype=np.int64))

    total_nll_nats = 0.0
    scored_events = 0
    scored_bytes = 0.0

    def score(contexts: torch.Tensor, targets: torch.Tensor) -> None:
        nonlocal total_nll_nats, scored_events
        with torch.no_grad():
            logits, _ = model(contexts.to(device))
            log_probabilities = F.log_softmax(logits[:, -1, :], dim=-1)
            nll = -log_probabilities.gather(1, targets.to(device)[:, None]).squeeze(1)
        total_nll_nats += float(nll.sum().item())
        scored_events += int(targets.numel())

    if args.evaluation_mode == "full-context":
        initial_stop = min(block_size - 1, len(ids_tensor) - 1)
        for target_index in range(1, initial_stop + 1):
            score(ids_tensor[:target_index].unsqueeze(0), ids_tensor[target_index:target_index + 1])
            scored_bytes += float(byte_weights[target_index].item())

        if len(ids_tensor) > block_size:
            contexts = ids_tensor.unfold(0, block_size, 1)
            targets = ids_tensor[block_size:]
            target_weights = byte_weights[block_size:]
            for offset in range(0, len(targets), args.evaluation_batch_size):
                target_batch = targets[offset:offset + args.evaluation_batch_size]
                score(contexts[offset:offset + len(target_batch)], target_batch)
                scored_bytes += float(target_weights[offset:offset + len(target_batch)].sum().item())
    else:
        windows = ids_tensor.unfold(0, block_size + 1, block_size)
        weight_windows = byte_weights.unfold(0, block_size + 1, block_size)
        for offset in range(0, len(windows), args.evaluation_batch_size):
            window_batch = windows[offset:offset + args.evaluation_batch_size]
            target_batch = window_batch[:, 1:]
            with torch.no_grad():
                logits, _ = model(
                    window_batch[:, :-1].to(device),
                    target_batch.to(device),
                )
                log_probabilities = F.log_softmax(logits, dim=-1)
                nll = -log_probabilities.gather(
                    2,
                    target_batch.to(device).unsqueeze(-1),
                ).squeeze(-1)
            total_nll_nats += float(nll.sum().item())
            scored_events += int(target_batch.numel())
            scored_bytes += float(weight_windows[offset:offset + len(window_batch), 1:].sum().item())

    bits = total_nll_nats / np.log(2.0)
    metrics = {
        "nll_nats": total_nll_nats,
        "event_unit": event_name,
        "evaluation_mode": args.evaluation_mode,
        "scored_events": scored_events,
        "scored_utf8_bytes": scored_bytes,
        "bits_per_utf8_byte": bits / scored_bytes,
        "nll_nats_per_event": total_nll_nats / scored_events,
    }
    report = {
        "schema_version": 1,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint_path).read_bytes()).hexdigest(),
        "test_text": args.test_text,
        "test_text_sha256": hashlib.sha256(raw_test_bytes).hexdigest(),
        "data_dir": args.data_dir,
        "device": str(device),
        "checkpoint": {
            "iteration": int(checkpoint["iter_num"]),
            "best_validation_loss": float(checkpoint["best_val_loss"]),
            "model_args": checkpoint["model_args"],
            "training_config": checkpoint["config"],
        },
        "teacher_forced": metrics,
    }
    output_path = Path(args.output_file) if args.output_file else Path(args.checkpoint_path).with_suffix(".test_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Test BPB: {metrics['bits_per_utf8_byte']:.6f}")
    print(f"Scored {event_name} events: {scored_events}")
    print(f"Scored UTF-8 bytes: {scored_bytes:.0f}")
    print(f"Wrote evaluation report to {output_path}")


if __name__ == "__main__":
    main()
