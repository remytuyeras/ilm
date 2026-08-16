"""
Interactive Hugging Face reference-model sandbox.

This is the reference-model counterpart to `sandbox/sandbox.py`. It loads one
Hugging Face causal language model once, then opens an interactive prompt loop.
It intentionally lives in `comparisons/` so it does not change ILM training or
the comparison-report harness.
"""

from __future__ import annotations

import argparse
import sys
from threading import Thread
from typing import Optional, Sequence

import torch


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"

DEFAULT_HF_CACHE_DIR = "comparisons/hf_cache"
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 3
DEFAULT_STREAM = True
DEFAULT_HF_REFERENCE = "karpathy-gpt2"
HF_REFERENCE_MODELS = {
    "karpathy-gpt2": {
        "model": "gpt2",
        "description": "OpenAI GPT-2 small loaded through the same Transformers path used by nanoGPT.",
    },
    "shakespeare-gpt2": {
        "model": "Esmaelmoat/SHAKESPEARE_GPT2",
        "description": "Community GPT-2-style model tuned for Shakespeare-like text.",
    },
    "sadia-gpt2-shakespeare": {
        "model": "sadia72/gpt2-shakespeare",
        "description": "GPT-2 fine-tuned on a Project Gutenberg Shakespeare corpus.",
    },
    "tinyshakespeare-42m": {
        "model": "MolecularReality/tinyshakespeare-42m",
        "description": "Larger GPT-2-style Tiny Shakespeare reference with roughly 30.5M parameters.",
    },
    "fawern-gpt2-shakespeare": {
        "model": "fawern/gpt2-shakespeare-text-generation",
        "tokenizer": "gpt2",
        "description": "GPT-2 small Shakespeare fine-tune with roughly 0.1B parameters.",
    },
}


def color_text(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_hf_model(args: argparse.Namespace) -> str:
    if args.hf_model:
        return args.hf_model
    return HF_REFERENCE_MODELS[args.hf_reference]["model"]


def resolve_hf_tokenizer(args: argparse.Namespace) -> str:
    if args.hf_tokenizer:
        return args.hf_tokenizer
    if args.hf_model:
        return args.hf_model
    return HF_REFERENCE_MODELS[args.hf_reference].get("tokenizer", resolve_hf_model(args))


def print_config(args: argparse.Namespace, device: torch.device) -> None:
    rows = [
        ("Reference", args.hf_reference),
        ("Model", resolve_hf_model(args)),
        ("Tokenizer", resolve_hf_tokenizer(args)),
        ("Device", str(device)),
        ("Max new tokens", args.max_new_tokens),
        ("Temperature", args.temperature),
        ("Top-k", args.top_k),
        ("Streaming", args.stream),
    ]
    title_width = max(len("Setting"), *(len(str(title)) for title, _ in rows))
    value_width = max(len("Value"), *(len(str(value)) for _, value in rows))
    rule = f"+-{'-' * title_width}-+-{'-' * value_width}-+"

    print(color_text("Reference model configuration", ANSI_BOLD + ANSI_CYAN))
    print(color_text(rule, ANSI_CYAN))
    print(color_text(f"| {'Setting'.ljust(title_width)} | {'Value'.ljust(value_width)} |", ANSI_BOLD))
    print(color_text(rule, ANSI_CYAN))
    for title, value in rows:
        print(f"| {str(title).ljust(title_width)} | {color_text(str(value).ljust(value_width), ANSI_GREEN)} |")
    print(color_text(rule, ANSI_CYAN))


def load_reference_model(args: argparse.Namespace, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "The Hugging Face sandbox needs optional dependencies. "
            "Run `pip install -r comparisons/requirements.txt`."
        ) from exc

    resolved_model = resolve_hf_model(args)
    resolved_tokenizer = resolve_hf_tokenizer(args)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_tokenizer,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generation_kwargs(args: argparse.Namespace, tokenizer) -> dict:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if args.temperature == 0:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = args.temperature
        if args.top_k is not None:
            kwargs["top_k"] = args.top_k
    return kwargs


def generate_text(prompt: str, tokenizer, model, args: argparse.Namespace, device: torch.device) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            **generation_kwargs(args, tokenizer),
        )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt):]
    return text


def stream_text(prompt: str, tokenizer, model, args: argparse.Namespace, device: torch.device) -> None:
    from transformers import TextIteratorStreamer

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    kwargs = {
        **encoded,
        **generation_kwargs(args, tokenizer),
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    for text in streamer:
        print(text, end="", flush=True)
    thread.join()
    print()


def interactive_loop(tokenizer, model, args: argparse.Namespace, device: torch.device) -> None:
    while True:
        prompt = input(color_text(">>> ", ANSI_BOLD + ANSI_CYAN))
        if prompt == "!exit":
            break
        if prompt == "!config":
            print_config(args, device)
            continue
        if not prompt:
            continue

        try:
            if args.stream:
                stream_text(prompt, tokenizer, model, args, device)
            else:
                print(generate_text(prompt, tokenizer, model, args, device))
        except KeyboardInterrupt:
            print()
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interact with a Hugging Face reference model.")
    parser.add_argument(
        "--hf-reference",
        choices=sorted(HF_REFERENCE_MODELS),
        default=DEFAULT_HF_REFERENCE,
        help="Named Hugging Face reference preset. Ignored when --hf-model is provided.",
    )
    parser.add_argument("--hf-model", default=None, help="Custom Hugging Face model id. Overrides --hf-reference.")
    parser.add_argument("--hf-tokenizer", default=None, help="Custom Hugging Face tokenizer id. Defaults to the resolved HF model.")
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--stream", action="store_true", default=DEFAULT_STREAM)
    parser.add_argument("--no-stream", action="store_false", dest="stream")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be zero or positive")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    device = get_device(args.device)
    tokenizer, model = load_reference_model(args, device)
    print_config(args, device)
    interactive_loop(tokenizer, model, args, device)


if __name__ == "__main__":
    main(sys.argv[1:])
