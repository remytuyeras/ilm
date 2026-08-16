"""
Compare ILM checkpoint generations against Hugging Face reference models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, ROOT_DIR)

import torch
from tqdm import tqdm

import ilm


DEFAULT_TOKENIZER_JSON = "data/tokenizer_embedding_cluster_v1.json"
DEFAULT_SYLLABLE_NUM = 3
DEFAULT_WORD_BLOCK_SIZE = 20
DEFAULT_VOCAB_SIZE = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_EMBEDDING_DIM = 300
DEFAULT_HEAD_NUM = 4
DEFAULT_HEAD_SIZE = DEFAULT_EMBEDDING_DIM // DEFAULT_HEAD_NUM
DEFAULT_LAYER_NUM = 6
DEFAULT_DROPOUT = 0.5
DEFAULT_COMPLETED_WORDS = 300
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 3
DEFAULT_STREAM = False
DEFAULT_PROMPTS_FILE = "comparisons/prompts/shakespeare.txt"
DEFAULT_OUTPUT_DIR = "comparisons/outputs"
DEFAULT_HF_MAX_NEW_TOKENS = 300
DEFAULT_HF_CACHE_DIR = "comparisons/hf_cache"
DEFAULT_HF_REFERENCE = "karpathy-gpt2"
DEFAULT_PROGRESS = True
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


def parse_int_sequence(value: str) -> Optional[Sequence[int]]:
    if value.strip().lower() in {"none", "off"}:
        return None
    try:
        items = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers, like 3,4,6, or none") from exc
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return items


def parse_float_sequence(value: str) -> Optional[Sequence[float]]:
    if value.strip().lower() in {"none", "off"}:
        return None
    try:
        items = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers, like 1,0.95,0.8, or none") from exc
    if any(item < 0 for item in items):
        raise argparse.ArgumentTypeError("values must be zero or positive")
    return items


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_prompts(prompt_values: Optional[List[str]], prompts_file: str) -> List[str]:
    prompts: List[str] = []
    if prompts_file:
        with open(prompts_file, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if line and not line.lstrip().startswith("#"):
                    prompts.append(line)

    if prompt_values:
        prompts.extend(prompt_values)

    if not prompts:
        raise ValueError("provide at least one prompt or a non-empty prompts file")

    return prompts


def format_sequence(value: Optional[Sequence[object]]) -> Optional[List[object]]:
    if value is None:
        return None
    return list(value)


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


def ilm_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "ilm_model_path": args.ilm_model_path,
        "tokenizer_json": args.tokenizer_json,
        "vocab_size": args.vocab_size,
        "block_size": args.block_size,
        "word_block_size": args.word_block_size,
        "syllable_num": args.syllable_num,
        "embedding_dim": args.embedding_dim,
        "head_num": args.head_num,
        "head_size": args.embedding_dim // args.head_num,
        "layer_num": args.layer_num,
        "dropout": args.dropout,
        "coordinate_token_embeddings": args.coordinate_token_embeddings,
        "coordinate_lm_heads": args.coordinate_lm_heads,
        "word_row_transformer": args.word_row_transformer,
        "completed_words": args.completed_words,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_k_by_coordinate": format_sequence(args.top_k_by_coordinate),
        "temperature_by_coordinate": format_sequence(args.temperature_by_coordinate),
    }


def hf_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "hf_reference": args.hf_reference,
        "hf_model": args.hf_model,
        "hf_tokenizer": args.hf_tokenizer,
        "resolved_hf_model": resolve_hf_model(args),
        "resolved_hf_tokenizer": resolve_hf_tokenizer(args),
        "hf_max_new_tokens": args.hf_max_new_tokens,
        "hf_temperature": args.hf_temperature,
        "hf_top_k": args.hf_top_k,
        "trust_remote_code": args.trust_remote_code,
    }


def build_ilm_generator(args: argparse.Namespace, device: torch.device) -> Callable[[str], str]:
    tokenizer, detokenizer = ilm.load_tokenizer(args.tokenizer_json)
    model = ilm.IntuinisticLanguageModel(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        block_size=args.block_size,
        layer_num=args.layer_num,
        device=device,
        dropout=args.dropout,
        syllable_num=args.syllable_num,
        word_block_size=args.word_block_size,
        head_num=args.head_num,
        coordinate_token_embeddings=args.coordinate_token_embeddings,
        coordinate_lm_heads=args.coordinate_lm_heads or args.word_row_transformer,
        word_row_transformer=args.word_row_transformer,
    )
    model.load_model(args.ilm_model_path)

    def generate(prompt: str) -> str:
        single_context = ilm.format_context(prompt, tokenizer=tokenizer).unsqueeze(0)
        generated_tokens = model.generate(
            single_context,
            max_new_tokens=args.syllable_num * args.completed_words,
            temperature=args.temperature,
            top_k=args.top_k,
            syllable_num=args.syllable_num,
            top_k_by_coordinate=args.top_k_by_coordinate,
            temperature_by_coordinate=args.temperature_by_coordinate,
            show_progress=False,
        ).detach().cpu()[0].tolist()
        token_codes = ilm.gather_tokens(generated_tokens, syllable_num=args.syllable_num)
        text = "".join(str(item) for item in detokenizer(token_codes))
        return text.replace(prompt, "", 1)

    return generate


def build_hf_generator(args: argparse.Namespace, device: torch.device) -> Callable[[str], str]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "The Hugging Face backend needs optional dependencies. "
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

    def generate(prompt: str) -> str:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs = {
            "max_new_tokens": args.hf_max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if args.hf_temperature == 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": args.hf_temperature,
                }
            )
            if args.hf_top_k is not None:
                generation_kwargs["top_k"] = args.hf_top_k

        with torch.no_grad():
            generated = model.generate(**encoded, **generation_kwargs)

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        if text.startswith(prompt):
            return text[len(prompt):]
        return text

    return generate


def run_backend(
        name: str,
        generator: Callable[[str], str],
        prompts: Sequence[str],
        config: Dict[str, object],
        show_progress: bool = DEFAULT_PROGRESS,
        ) -> List[Dict[str, object]]:
    records = []
    iterator = tqdm(
        prompts,
        desc=f"{name} generation",
        unit="prompt",
        disable=not show_progress,
    )
    for prompt in iterator:
        started = time.time()
        try:
            completion = generator(prompt)
            error = None
        except Exception as exc:
            completion = ""
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - started
        records.append(
            {
                "backend": name,
                "prompt": prompt,
                "completion": completion,
                "error": error,
                "elapsed_seconds": elapsed,
                "config": config,
            }
        )
    return records


def backend_load_error_records(
        name: str,
        error: Exception,
        prompts: Iterable[str],
        config: Dict[str, object],
        ) -> List[Dict[str, object]]:
    message = f"{type(error).__name__}: {error}"
    return [
        {
            "backend": name,
            "prompt": prompt,
            "completion": "",
            "error": message,
            "elapsed_seconds": 0.0,
            "config": config,
        }
        for prompt in prompts
    ]


def write_reports(records: List[Dict[str, object]], output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(output_dir, f"comparison_{timestamp}.json")
    md_path = os.path.join(output_dir, f"comparison_{timestamp}.md")

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
        f.write("\n")

    with open(md_path, "w") as f:
        f.write("# Generation Comparison\n\n")
        f.write(f"Created at: `{timestamp}`\n\n")
        for record in records:
            f.write(f"## {record['backend']} | {record['prompt']}\n\n")
            f.write(f"Elapsed: `{record['elapsed_seconds']:.3f}s`\n\n")
            if record["error"]:
                f.write(f"Error: `{record['error']}`\n\n")
                continue
            f.write("```text\n")
            f.write(record["completion"])
            f.write("\n```\n\n")

    return {"json": json_path, "markdown": md_path}


def print_terminal_report(records: List[Dict[str, object]]) -> None:
    for record in records:
        print("=" * 80)
        print(f"{record['backend']} | prompt: {record['prompt']}")
        print(f"elapsed: {record['elapsed_seconds']:.3f}s")
        if record["error"]:
            print(f"error: {record['error']}")
        else:
            print(record["completion"])
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare ILM output against reference language models.")
    parser.add_argument("--backend", choices=["ilm", "hf", "both"], default="both")
    parser.add_argument("--prompt", action="append", help="Prompt to run. Can be provided multiple times.")
    parser.add_argument("--prompts-file", default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--ilm-model-path")
    parser.add_argument("--tokenizer-json", default=DEFAULT_TOKENIZER_JSON)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--word-block-size", type=int, default=DEFAULT_WORD_BLOCK_SIZE)
    parser.add_argument("--syllable-num", type=int, default=DEFAULT_SYLLABLE_NUM)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--head-num", type=int, default=DEFAULT_HEAD_NUM)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--coordinate-token-embeddings", action="store_true")
    parser.add_argument("--coordinate-lm-heads", action="store_true")
    parser.add_argument("--word-row-transformer", action="store_true")
    parser.add_argument("--completed-words", type=int, default=DEFAULT_COMPLETED_WORDS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-k-by-coordinate", type=parse_int_sequence, default=None)
    parser.add_argument("--temperature-by-coordinate", type=parse_float_sequence, default=None)

    parser.add_argument(
        "--hf-reference",
        choices=sorted(HF_REFERENCE_MODELS),
        default=DEFAULT_HF_REFERENCE,
        help="Named Hugging Face reference preset. Ignored when --hf-model is provided.",
    )
    parser.add_argument("--hf-model", help="Custom Hugging Face model id. Overrides --hf-reference.")
    parser.add_argument("--hf-tokenizer", help="Custom Hugging Face tokenizer id. Defaults to the resolved HF model.")
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--hf-max-new-tokens", type=int, default=DEFAULT_HF_MAX_NEW_TOKENS)
    parser.add_argument("--hf-temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--hf-top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--progress", action="store_true", default=DEFAULT_PROGRESS)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.block_size is None:
        args.block_size = args.syllable_num * args.word_block_size
    if args.block_size != args.syllable_num * args.word_block_size:
        parser.error("--block-size must equal --syllable-num * --word-block-size")
    if args.head_num <= 0:
        parser.error("--head-num must be positive")
    if args.embedding_dim % args.head_num != 0:
        parser.error("--embedding-dim must be divisible by --head-num")
    if args.word_row_transformer:
        args.coordinate_lm_heads = True
    if args.backend in {"ilm", "both"} and not args.ilm_model_path:
        parser.error("--ilm-model-path is required when --backend is ilm or both")
    if args.backend in {"hf", "both"} and not (args.hf_model or args.hf_reference):
        parser.error("--hf-reference or --hf-model is required when --backend is hf or both")
    if args.top_k_by_coordinate is not None and len(args.top_k_by_coordinate) != args.syllable_num:
        parser.error("--top-k-by-coordinate must have one value per syllable")
    if args.temperature_by_coordinate is not None and len(args.temperature_by_coordinate) != args.syllable_num:
        parser.error("--temperature-by-coordinate must have one value per syllable")
    if args.temperature < 0 or args.hf_temperature < 0:
        parser.error("temperature values must be zero or positive")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.hf_top_k is not None and args.hf_top_k <= 0:
        parser.error("--hf-top-k must be positive")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    prompts = load_prompts(args.prompt, args.prompts_file)
    device = get_device(args.device)
    records: List[Dict[str, object]] = []

    if args.backend in {"ilm", "both"}:
        config = ilm_config(args)
        try:
            print("Loading ILM backend...")
            ilm_generator = build_ilm_generator(args, device)
            records.extend(run_backend("ilm", ilm_generator, prompts, config, args.progress))
        except Exception as exc:
            records.extend(backend_load_error_records("ilm", exc, prompts, config))

    if args.backend in {"hf", "both"}:
        config = hf_config(args)
        try:
            print("Loading Hugging Face backend...")
            hf_generator = build_hf_generator(args, device)
            records.extend(run_backend("huggingface", hf_generator, prompts, config, args.progress))
        except Exception as exc:
            records.extend(backend_load_error_records("huggingface", exc, prompts, config))

    print_terminal_report(records)
    paths = write_reports(records, args.output_dir)
    print(f"Wrote JSON report to {paths['json']}")
    print(f"Wrote Markdown report to {paths['markdown']}")


if __name__ == "__main__":
    main()
