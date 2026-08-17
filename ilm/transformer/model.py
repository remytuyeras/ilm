from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, List, Sequence, Tuple
from tqdm import trange
import math
import os
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from ..tokenizer.core import find_tokens


def set_seed(seed: int = 42) -> None:
    """Reset the random generators used by ILM training and sampling."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class EncodingStats:
    source_bytes: int
    tokenized_source_bytes: int
    token_count: int
    oov_token_count: int
    coordinate_count: int
    fallback_code: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_bytes": self.source_bytes,
            "tokenized_source_bytes": self.tokenized_source_bytes,
            "token_count": self.token_count,
            "oov_token_count": self.oov_token_count,
            "oov_rate": self.oov_token_count / self.token_count if self.token_count else 0.0,
            "coordinate_count": self.coordinate_count,
            "fallback_code": self.fallback_code,
        }


class UnknownTokenError(ValueError):
    """Raised when a closed-vocabulary tokenizer meets an out-of-vocabulary token."""

    def __init__(self, stats: EncodingStats, examples: Sequence[str]):
        self.stats = stats
        self.examples = list(examples)
        super().__init__(
            "Tokenizer encountered "
            f"{stats.oov_token_count}/{stats.token_count} out-of-vocabulary tokens "
            f"({stats.oov_token_count / stats.token_count:.2%}): {self.examples!r}. "
            "Use oov_policy='fallback' only when that deterministic fallback is part of the experiment."
        )


def _code_sort_key(code: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in code.split(":"))


def _resolve_fallback_code(
        tokenizer: Callable[[str], List[Optional[str]]],
        observed_codes: Sequence[Optional[str]],
        fallback_code: Optional[str],
        ) -> str:
    if fallback_code is not None:
        return fallback_code

    direct_mapping = getattr(tokenizer, "direct_mapping", None)
    available_codes = direct_mapping.values() if direct_mapping is not None else observed_codes
    candidates = sorted({code for code in available_codes if code is not None}, key=_code_sort_key)
    if not candidates:
        raise ValueError("cannot choose an OOV fallback code from an empty tokenizer mapping")
    return candidates[0]


def encode_context(
        raw_text: str,
        tokenizer: Callable[[str], List[Optional[str]]],
        oov_policy: str = "error",
        fallback_code: Optional[str] = None,
        ) -> Tuple[torch.Tensor, EncodingStats, torch.Tensor]:
    """Encode text and retain OOV and source-byte accounting for experiments."""
    if oov_policy not in {"error", "fallback"}:
        raise ValueError("oov_policy must be 'error' or 'fallback'")

    text_tokens = find_tokens(
        raw_text,
        lossless=bool(getattr(tokenizer, "lossless_tokenization", False)),
    )
    mapped_codes = tokenizer(raw_text)
    if len(text_tokens) != len(mapped_codes):
        raise ValueError("tokenizer output length does not match the source tokenization")

    oov_examples = [token for token, code in zip(text_tokens, mapped_codes) if code is None]
    resolved_fallback = None
    if oov_examples and oov_policy == "fallback":
        resolved_fallback = _resolve_fallback_code(tokenizer, mapped_codes, fallback_code)

    coordinates: List[int] = []
    coordinate_byte_weights: List[float] = []
    tokenized_source_bytes = 0
    for token, code in zip(text_tokens, mapped_codes):
        token_bytes = len(token.encode("utf-8"))
        tokenized_source_bytes += token_bytes
        if code is None:
            code = resolved_fallback
        if code is None:
            continue
        parts = code.split(":")
        try:
            coordinates.extend(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"tokenizer emitted a non-numeric code: {code!r}") from exc
        coordinate_byte_weights.extend([token_bytes / len(parts)] * len(parts))

    stats = EncodingStats(
        source_bytes=len(raw_text.encode("utf-8")),
        tokenized_source_bytes=tokenized_source_bytes,
        token_count=len(text_tokens),
        oov_token_count=len(oov_examples),
        coordinate_count=len(coordinates),
        fallback_code=resolved_fallback,
    )
    if oov_examples and oov_policy == "error":
        raise UnknownTokenError(stats, oov_examples[:8])
    if not coordinates:
        raise ValueError("text did not produce any tokenizer coordinates")

    return (
        torch.tensor(coordinates, dtype=torch.long),
        stats,
        torch.tensor(coordinate_byte_weights, dtype=torch.float64),
    )


def format_context(
        raw_text: str,
        tokenizer: Callable[[str], List[Optional[str]]],
        oov_policy: str = "error",
        fallback_code: Optional[str] = None,
        return_stats: bool = False,
        ):
    """Return the flattened coordinate stream, optionally with encoding statistics."""
    context, stats, coordinate_byte_weights = encode_context(
        raw_text=raw_text,
        tokenizer=tokenizer,
        oov_policy=oov_policy,
        fallback_code=fallback_code,
    )
    if return_stats:
        return context, stats, coordinate_byte_weights
    return context

def gather_tokens(tokens: List[int], syllable_num: int = 3) -> List[str]:
    output = []

    # prepare the parsing of (syllable_num)-tokens
    quotient = len(tokens) // syllable_num
    remainder = len(tokens) % syllable_num
    
    # convert single_context into list of (syllable_num)-tokens
    for i in range(quotient):
        output.append(":".join([str(tokens[syllable_num*i+j]) for j in range(syllable_num)]))
    
    # handle potentially incomplete (syllable_num)-tokens
    if remainder > 0:
        output.append(":".join([str(tokens[x]) for x in range(syllable_num * quotient, len(tokens))] + ["?"] * remainder))
    
    return output


def build_word_prefix_alignment(
        start_positions: torch.Tensor,
        block_size: int,
        syllable_num: int,
        word_block_size: int,
        device=torch.device("cpu"),
        ):
    if syllable_num <= 0:
        raise ValueError("syllable_num must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if word_block_size <= 0:
        raise ValueError("word_block_size must be positive")

    start_positions = start_positions.to(device=device, dtype=torch.long)
    local_positions = torch.arange(block_size, device=device)
    corpus_row_indices = (start_positions[:, None] + local_positions[None, :]) // syllable_num
    max_row_indices = corpus_row_indices[:, -1]
    row_counts = torch.minimum(max_row_indices + 1, torch.full_like(max_row_indices, word_block_size))
    row_offsets = torch.arange(word_block_size, device=device)
    first_row_indices = max_row_indices - row_counts + 1
    selected_rows = first_row_indices[:, None] + row_offsets[None, :]
    row_is_active = row_offsets[None, :] < row_counts[:, None]

    J_x = (
        corpus_row_indices[:, None, :] == selected_rows[:, :, None]
    ).float()
    J_x = J_x * row_is_active[:, :, None].float()

    return J_x


class TrainingManager(object):

    def __init__(
            self,
            raw_text,
            tokenizer,
            device=torch.device("cpu"),
            batch_size: int = 32,
            block_size: int = 3 * 8,
            syllable_num: int = 3,
            return_start_positions: bool = False,
            validation_text: Optional[str] = None,
            test_text: Optional[str] = None,
            oov_policy: str = "error",
            fallback_code: Optional[str] = None,
            ):
        if syllable_num <= 0:
            raise ValueError("syllable_num must be positive")
        if (validation_text is None) != (test_text is None):
            raise ValueError("validation_text and test_text must be provided together")

        self.batch_size = batch_size
        self.block_size = block_size
        self.syllable_num = syllable_num
        self.return_start_positions = return_start_positions
        self.device = device
        self.tokenizer = tokenizer
        self.oov_policy = oov_policy
        self.fallback_code = fallback_code
        self.training_coordinate_events_consumed = 0
        self.training_source_bytes_observed = 0.0

        if validation_text is None:
            self.split_mode = "implicit_80_20"
            text_tokens, text_stats, byte_weights = self._encode(raw_text)
            self.source_encoding_stats = text_stats
            self.text_tokens = text_tokens
            self.n_dataset = len(text_tokens)
            self.n_training = int(0.8 * self.n_dataset)
            self._set_split(
                "train",
                text_tokens[:self.n_training],
                byte_weights[:self.n_training],
                self._implicit_split_stats("train", text_tokens[:self.n_training], byte_weights[:self.n_training]),
                offset=0,
            )
            self._set_split(
                "validate",
                text_tokens[self.n_training:],
                byte_weights[self.n_training:],
                self._implicit_split_stats(
                    "validate",
                    text_tokens[self.n_training:],
                    byte_weights[self.n_training:],
                ),
                offset=self.n_training,
            )
            print("n_dataset =", self.n_dataset)
            print("n_training =", self.n_training)
        else:
            self.split_mode = "explicit_train_validation_test"
            train_tokens, train_stats, train_weights = self._encode(raw_text)
            validation_tokens, validation_stats, validation_weights = self._encode(validation_text)
            test_tokens, test_stats, test_weights = self._encode(test_text)
            self.text_tokens = train_tokens
            self.n_dataset = len(train_tokens) + len(validation_tokens) + len(test_tokens)
            self.n_training = len(train_tokens)
            self._set_split("train", train_tokens, train_weights, train_stats, offset=0)
            self._set_split("validate", validation_tokens, validation_weights, validation_stats, offset=0)
            self._set_split("test", test_tokens, test_weights, test_stats, offset=0)
            print("n_training =", len(train_tokens))
            print("n_validation =", len(validation_tokens))
            print("n_test =", len(test_tokens))

    def _encode(self, raw_text: str):
        return format_context(
            raw_text=raw_text,
            tokenizer=self.tokenizer,
            oov_policy=self.oov_policy,
            fallback_code=self.fallback_code,
            return_stats=True,
        )

    @staticmethod
    def _implicit_split_stats(
            name: str,
            coordinates: torch.Tensor,
            coordinate_byte_weights: torch.Tensor,
            ) -> Dict[str, Any]:
        """Describe a coordinate slice without inventing token-level OOV counts."""
        return {
            "coordinate_count": len(coordinates),
            "source_bytes_estimate": float(coordinate_byte_weights.sum().item()),
            "oov_statistics": "reported for the combined source_encoding only",
            "split_name": name,
        }

    def _set_split(
            self,
            name: str,
            coordinates: torch.Tensor,
            coordinate_byte_weights: torch.Tensor,
            stats: Any,
            offset: int,
            ) -> None:
        if not hasattr(self, "split_data"):
            self.split_data = {}
            self.split_offsets = {}
            self.split_byte_weights = {}
            self.split_stats = {}
        self.split_data[name] = coordinates.to(self.device)
        self.split_offsets[name] = offset
        self.split_byte_weights[name] = coordinate_byte_weights.cpu()
        self.split_stats[name] = stats

        if name == "train":
            self.training_data = self.split_data[name]
        elif name == "validate":
            self.validation_data = self.split_data[name]
        elif name == "test":
            self.test_data = self.split_data[name]

    def data_statistics(self) -> Dict[str, Any]:
        statistics = {
            "split_mode": self.split_mode,
            "oov_policy": self.oov_policy,
            "splits": {
                name: stats.to_dict() if isinstance(stats, EncodingStats) else stats
                for name, stats in self.split_stats.items()
            },
            "training_coordinate_events_consumed": self.training_coordinate_events_consumed,
            "training_source_bytes_observed": self.training_source_bytes_observed,
        }
        if self.split_mode == "implicit_80_20":
            statistics["source_encoding"] = self.source_encoding_stats.to_dict()
        return statistics

    def get_batch(self, split: str, track_consumption: bool = False):
        if split == "validation":
            split = "validate"
        if split not in self.split_data:
            available = ", ".join(sorted(self.split_data))
            raise ValueError(f"unknown split {split!r}; available splits: {available}")

        data = self.split_data[split]
        data_offset = self.split_offsets[split]

        max_index = len(data) - self.block_size
        if max_index <= 0:
            raise ValueError("dataset split is too small for the configured block_size")

        # random indices accross document
        indices = torch.randint(max_index, (self.batch_size,))

        # size (B, T) = Batch dimension, Time dimension
        x = torch.stack([data[i:i+self.block_size] for i in indices])
        y = torch.stack([data[i+1:i+1+self.block_size] for i in indices])

        if track_consumption:
            byte_weights = self.split_byte_weights[split]
            self.training_coordinate_events_consumed += self.batch_size * self.block_size
            self.training_source_bytes_observed += sum(
                float(byte_weights[int(index):int(index) + self.block_size].sum().item())
                for index in indices
            )

        if self.return_start_positions:
            absolute_indices = data_offset + indices
            start_positions = (absolute_indices % self.syllable_num).to(self.device)
            return x, y, start_positions

        return x, y
    
    def format_output(self, tokens: List[int], syllable_num: int = 3) -> List[str]:
        return gather_tokens(tokens=tokens, syllable_num=syllable_num)
    
    def format_input(self, raw_text: str) -> torch.Tensor:
        return format_context(
            raw_text=raw_text,
            tokenizer=self.tokenizer,
            oov_policy=self.oov_policy,
            fallback_code=self.fallback_code,
        )


class ILMHead(nn.Module):
    
    def __init__(self,
                 embedding_dim=32, # channels
                 block_size=12, # time
                 head_size=16,
                 device=torch.device("cpu"),
                 dropout=0.2):
        
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.device = device

        self.key = nn.Linear(embedding_dim, head_size, bias=False).to(self.device)
        self.query = nn.Linear(embedding_dim,head_size, bias=False).to(self.device)
        self.value = nn.Linear(embedding_dim, head_size, bias=False).to(self.device)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).to(self.device))

        self.dropout = nn.Dropout(dropout)

    def forward(self, batched_context: torch.Tensor):
        T = batched_context.shape[1]

        q: torch.Tensor = self.query(batched_context) # (B, T, H)
        k: torch.Tensor = self.key(batched_context) # (B, T, H)
        v: torch.Tensor = self.value(batched_context) # (B, T, H)

        weights: torch.Tensor = q @ k.transpose(-2,-1) / (self.head_size ** 0.5) # (B, T, T)

        weights = weights.masked_fill(
            self.tril[:T,:T] == 0,
            float("-inf")) # Decoder structure
        
        weights = F.softmax(weights, dim=-1) # (B, T)
        weights = self.dropout(weights)
        output = weights @ v # (B, H)

        return output

class ILMMultiHead(nn.Module):
    
    def __init__(self,
                 head_num=4, 
                 embedding_dim=32, # channels
                 block_size=12, # time
                 head_size=16,
                 device=torch.device("cpu"),
                 dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([ILMHead(embedding_dim, block_size, head_size, device, dropout) for _ in range(head_num)]) 
        self.proj = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            input_emb,
            ):
        emb = torch.cat([h(input_emb) for h in self.heads], dim=-1)
        emb = self.proj(emb)
        emb = self.dropout(emb)
        return emb


class ILMFeedForward(nn.Module):
    
    def __init__(self, embedding_dim=32, device=torch.device("cpu"), dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
            ).to(device)
        
    def forward(self,batched_embeddings):
        return self.net(batched_embeddings)

class ILMBlock(nn.Module):
    
    def __init__(self, 
                 head_num=4,
                 embedding_dim=32, 
                 block_size=12,
                 device=torch.device("cpu"),
                 dropout=0.2):
        super().__init__()
        head_size = embedding_dim // head_num
        self.sa_heads = ILMMultiHead(head_num=head_num,
                                     embedding_dim=embedding_dim, 
                                     block_size=block_size,
                                     head_size=head_size,
                                     device=device,
                                     dropout=dropout)
        self.ffwrd = ILMFeedForward(embedding_dim=embedding_dim, device=device, dropout=dropout)
        self.ln1 = nn.LayerNorm(embedding_dim).to(device)
        self.ln2 = nn.LayerNorm(embedding_dim).to(device)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwrd(self.ln2(x))
        return x

class IntuinisticLanguageModel(nn.Module):
    
    def __init__(self, 
                 vocab_size=64, # vocabulary
                 embedding_dim=32, # channels
                 block_size=12, # time
                 head_size=16,
                 layer_num=4,
                 device=torch.device("cpu"),
                 dropout=0.2,
                 syllable_num: int = 3,
                 word_block_size: Optional[int] = None,
                 head_num: int = 4,
                 ilm_input_embeddings: bool = False,
                 ilm_output_heads: bool = False,
                 ilm_objective: bool = False):
        
        super().__init__()
        if syllable_num <= 0:
            raise ValueError("syllable_num must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if head_num <= 0:
            raise ValueError("head_num must be positive")
        if embedding_dim % head_num != 0:
            raise ValueError("embedding_dim must be divisible by head_num")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.head_num = head_num
        # Head width follows the model width and number of attention heads.
        self.head_size = embedding_dim // head_num
        self.syllable_num = syllable_num
        self.ilm_input_embeddings = ilm_input_embeddings
        self.ilm_output_heads = ilm_output_heads
        self.ilm_objective = ilm_objective

        if word_block_size is None and block_size % syllable_num == 0:
            word_block_size = block_size // syllable_num
        if ilm_objective:
            if block_size % syllable_num != 0:
                raise ValueError("ilm_objective=True requires block_size to be divisible by syllable_num")
            expected_word_block_size = block_size // syllable_num
            if word_block_size != expected_word_block_size:
                raise ValueError(
                    "ilm_objective=True requires word_block_size to equal "
                    "block_size // syllable_num"
                )
        self.word_block_size = word_block_size

        self.device=device
        
        token_embedding_vocab_size = vocab_size
        if self.ilm_input_embeddings:
            token_embedding_vocab_size = syllable_num * vocab_size
        self.token_embedding_table = nn.Embedding(token_embedding_vocab_size, embedding_dim).to(self.device)
        self.pos_embedding_table = nn.Embedding(block_size, embedding_dim).to(self.device)

        # self.sa_head = ILMHead(
        #     embedding_dim=embedding_dim,
        #     block_size=block_size,
        #     head_size=embedding_dim,
        #     device=device)

        # self.sa_heads = ILMMultiHead(
        #     head_num=4,
        #     embedding_dim=embedding_dim,
        #     block_size=block_size,
        #     head_size=embedding_dim//4,
        #     device=device)
        
        # self.ffwrd = ILMFeedForward(embedding_dim=embedding_dim, device=device)

        self.blocks = nn.Sequential(
            *[ILMBlock(
                head_num=self.head_num,
                embedding_dim=embedding_dim,
                block_size=block_size,
                device=device,
                dropout=dropout)
            for _ in range(layer_num)])
        
        self.ln_f = nn.LayerNorm(embedding_dim).to(device)

        if self.ilm_output_heads:
            self.lm_heads = nn.ModuleList([
                nn.Linear(embedding_dim, vocab_size).to(self.device)
                for _ in range(syllable_num)
            ])
        else:
            self.lm_head = nn.Linear(embedding_dim, vocab_size).to(self.device)

        # print(self.token_embedding_table(torch.tensor([0,1], dtype=torch.long)))

    def _standard_loss(self, batched_logits: torch.Tensor, batched_targets: torch.Tensor):
        B, T, C = batched_logits.shape # Batch, Time, Channel
        # concatenate logits C-vectors ovar all B-batches
        flattened_logits = batched_logits.view(B * T, C).to(self.device)
        # concatenate token T-tuples over all B-batches
        flattened_targets = batched_targets.view(B * T).to(self.device)
        # compute cross_entropy loss where each entry of flattened_targets becomes the binary C-vector
        return F.cross_entropy(flattened_logits, flattened_targets)

    def _coordinate_target_roles(self, start_positions: torch.Tensor, T: int):
        if start_positions is None:
            raise ValueError("ilm_output_heads=True requires start_positions")
        start_positions = start_positions.to(self.device, dtype=torch.long)
        if start_positions.ndim != 1:
            raise ValueError("start_positions must have shape (B,)")
        positions = torch.arange(T, device=self.device)
        return (start_positions[:, None] + positions[None, :] + 1) % self.syllable_num

    def _coordinate_input_roles(self, start_positions: torch.Tensor, T: int):
        if start_positions is None:
            raise ValueError("ilm_input_embeddings=True requires start_positions")
        start_positions = start_positions.to(self.device, dtype=torch.long)
        if start_positions.ndim != 1:
            raise ValueError("start_positions must have shape (B,)")
        positions = torch.arange(T, device=self.device)
        return (start_positions[:, None] + positions[None, :]) % self.syllable_num

    def _token_embeddings(
            self,
            batched_context: torch.Tensor,
            start_positions: Optional[torch.Tensor],
            ):
        if not self.ilm_input_embeddings:
            return self.token_embedding_table(batched_context)

        input_roles = self._coordinate_input_roles(
            start_positions=start_positions,
            T=batched_context.shape[1],
        )
        role_token_indices = input_roles * self.vocab_size + batched_context
        return self.token_embedding_table(role_token_indices)

    def _coordinate_logits(self, batched_emb: torch.Tensor, start_positions: torch.Tensor):
        B, T, _ = batched_emb.shape
        target_roles = self._coordinate_target_roles(start_positions=start_positions, T=T)
        return self._coordinate_logits_for_roles(
            batched_emb=batched_emb,
            target_roles=target_roles,
        )

    def _coordinate_logits_for_roles(self, batched_emb: torch.Tensor, target_roles: torch.Tensor):
        B, T, _ = batched_emb.shape
        if target_roles.shape[0] != B:
            raise ValueError("target_roles must have one row per batch item")
        if target_roles.shape[1] != T:
            raise ValueError("target_roles must have one role per output position")

        logits_by_head = torch.stack(
            [head(batched_emb) for head in self.lm_heads],
            dim=2,
        )  # (B, T, S, V)
        head_index = target_roles[:, :, None, None].expand(-1, -1, 1, self.vocab_size)
        return logits_by_head.gather(dim=2, index=head_index).squeeze(2)

    def _word_prefix_loss(
            self,
            batched_logits: torch.Tensor,
            batched_targets: torch.Tensor,
            J_x: torch.Tensor,
            ):
        B, T, V = batched_logits.shape
        if batched_targets.shape[:2] != (B, T):
            raise ValueError("word-prefix loss requires targets with shape (B, T)")
        prefix_mask = J_x.bool().any(dim=1)
        valid_logits = batched_logits[prefix_mask]
        valid_targets = batched_targets[prefix_mask]
        if valid_logits.numel() == 0:
            raise ValueError("word-prefix loss has no valid targets")
        return F.cross_entropy(valid_logits.reshape(-1, V), valid_targets.reshape(-1))

    def forward(
            self,
            batched_context: torch.Tensor,
            batched_targets: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            ):
        # self.eval()
        
        T = batched_context.shape[1]
        # batched_context: (B, T) -> batched_tok_emb: (B, T, C)
        batched_tok_emb: torch.Tensor = self._token_embeddings(
            batched_context=batched_context,
            start_positions=start_positions,
        )
        # position vectors  -> batched_tok_emb: (T, C)
        batched_pos_emb: torch.Tensor = self.pos_embedding_table(torch.arange(T, device=self.device).to(self.device))
        # embedding and postiond 
        batched_emb = batched_tok_emb + batched_pos_emb

        # self attention
        batched_emb = self.blocks(batched_emb)
        batched_emb = self.ln_f(batched_emb)
        if self.ilm_output_heads:
            batched_logits = self._coordinate_logits(
                batched_emb=batched_emb,
                start_positions=start_positions,
            )
        else:
            batched_logits: torch.Tensor = self.lm_head(batched_emb)
        
        if batched_targets is None:
            loss = None

        else:
            if self.ilm_objective:
                if start_positions is None:
                    raise ValueError("ilm_objective=True requires start_positions")
                J_x = build_word_prefix_alignment(
                    start_positions=start_positions,
                    block_size=T,
                    syllable_num=self.syllable_num,
                    word_block_size=self.word_block_size,
                    device=self.device,
                )
                loss = self._word_prefix_loss(
                    batched_logits=batched_logits,
                    batched_targets=batched_targets,
                    J_x=J_x,
                )
            else:
                loss = self._standard_loss(
                    batched_logits=batched_logits,
                    batched_targets=batched_targets,
                )
        
        return batched_logits, loss

    def _unpack_training_batch(self, batch):
        if (
                self.ilm_input_embeddings
                or self.ilm_output_heads
                or self.ilm_objective
                ):
            if len(batch) != 3:
                raise ValueError(
                    "this model requires TrainingManager(return_start_positions=True)"
                )
            return batch

        x, y = batch[:2]
        return x, y, None
    

    @torch.no_grad()
    def estimate_loss(self, manager: TrainingManager):
        eval_iters = 100
        out = {}
        self.eval()
        for split in ["train", "validate"]:
            losses = torch.zeros(eval_iters).to(self.device)
            for k in range(eval_iters):
                x, y, start_positions = self._unpack_training_batch(manager.get_batch(split))
                _, loss = self(x, y, start_positions=start_positions); loss: torch.Tensor
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out

    @torch.no_grad()
    def evaluate_teacher_forced(
            self,
            coordinates: torch.Tensor,
            coordinate_byte_weights: torch.Tensor,
            batch_size: int = 32,
            evaluation_mode: str = "full-context",
            ) -> Dict[str, float]:
        """Evaluate autoregressive coordinate likelihood without sampling.

        ``full-context`` scores each target with up to ``block_size`` true
        preceding coordinates. ``block-reset`` scores each coordinate once in
        non-overlapping context blocks, matching the finite training windows.
        The word-prefix loss mask is deliberately not applied here.
        """
        if coordinates.ndim != 1:
            raise ValueError("coordinates must have shape (N,)")
        if coordinate_byte_weights.shape != coordinates.shape:
            raise ValueError("coordinate_byte_weights must match coordinates")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if evaluation_mode not in {"full-context", "block-reset"}:
            raise ValueError("evaluation_mode must be 'full-context' or 'block-reset'")
        if len(coordinates) < 2:
            raise ValueError("at least two coordinates are required for teacher-forced evaluation")

        was_training = self.training
        self.eval()
        total_nll_nats = 0.0
        coordinate_events = 0
        scored_source_bytes = 0.0
        coordinates_cpu = coordinates.detach().cpu()
        byte_weights_cpu = coordinate_byte_weights.detach().cpu().to(torch.float64)

        def score_contexts(contexts: torch.Tensor, targets: torch.Tensor, starts: torch.Tensor) -> None:
            nonlocal total_nll_nats, coordinate_events, scored_source_bytes
            start_positions = None
            if self.ilm_input_embeddings or self.ilm_output_heads:
                start_positions = starts.to(self.device)
            logits, _ = self(contexts.to(self.device), start_positions=start_positions)
            log_probabilities = F.log_softmax(logits[:, -1, :], dim=-1)
            nll = -log_probabilities.gather(1, targets.to(self.device)[:, None]).squeeze(1)
            total_nll_nats += float(nll.sum().item())
            coordinate_events += int(targets.numel())

        if evaluation_mode == "full-context":
            # The first targets have shorter available histories.
            initial_target_stop = min(self.block_size - 1, len(coordinates_cpu) - 1)
            for target_index in range(1, initial_target_stop + 1):
                score_contexts(
                    contexts=coordinates_cpu[:target_index].unsqueeze(0),
                    targets=coordinates_cpu[target_index:target_index + 1],
                    starts=torch.tensor([0]),
                )
                scored_source_bytes += float(byte_weights_cpu[target_index].item())

            # Thereafter each target has a full causal context of length block_size.
            if len(coordinates_cpu) > self.block_size:
                windows = coordinates_cpu.unfold(0, self.block_size, 1)
                targets = coordinates_cpu[self.block_size:]
                target_weights = byte_weights_cpu[self.block_size:]
                for offset in range(0, len(targets), batch_size):
                    target_batch = targets[offset:offset + batch_size]
                    context_batch = windows[offset:offset + len(target_batch)]
                    target_indices = torch.arange(
                        self.block_size + offset,
                        self.block_size + offset + len(target_batch),
                    )
                    starts = (target_indices - self.block_size) % self.syllable_num
                    score_contexts(context_batch, target_batch, starts)
                    scored_source_bytes += float(target_weights[offset:offset + len(target_batch)].sum().item())
        else:
            windows = coordinates_cpu.unfold(0, self.block_size + 1, self.block_size)
            weight_windows = byte_weights_cpu.unfold(0, self.block_size + 1, self.block_size)
            for offset in range(0, len(windows), batch_size):
                window_batch = windows[offset:offset + batch_size]
                context_batch = window_batch[:, :-1]
                target_batch = window_batch[:, 1:]
                starts = (
                    torch.arange(offset, offset + len(window_batch)) * self.block_size
                ) % self.syllable_num
                start_positions = starts.to(self.device) if (
                    self.ilm_input_embeddings or self.ilm_output_heads
                ) else None
                logits, _ = self(context_batch.to(self.device), start_positions=start_positions)
                log_probabilities = F.log_softmax(logits, dim=-1)
                nll = -log_probabilities.gather(
                    2,
                    target_batch.to(self.device).unsqueeze(-1),
                ).squeeze(-1)
                total_nll_nats += float(nll.sum().item())
                coordinate_events += int(target_batch.numel())
                scored_source_bytes += float(
                    weight_windows[offset:offset + len(window_batch), 1:].sum().item()
                )

        if was_training:
            self.train()
        bits = total_nll_nats / np.log(2.0)
        return {
            "coordinate_nll_nats": total_nll_nats,
            "coordinate_events": coordinate_events,
            "evaluation_mode": evaluation_mode,
            "scored_source_bytes": scored_source_bytes,
            "bits_per_utf8_byte": bits / scored_source_bytes,
            "coordinate_nll_nats_per_event": total_nll_nats / coordinate_events,
        }


    def train_model(
            self,
            manager: TrainingManager,
            epoch_num: int = 5000,
            lr: float = 1e-3,
            validation_interval: int = 500,
            weight_decay: float = 0.01,
            beta1: float = 0.9,
            beta2: float = 0.999,
            grad_clip: float = 0.0,
            optimizer_profile: str = "all-parameters",
            lr_schedule: str = "constant",
            warmup_iters: int = 0,
            lr_decay_iters: Optional[int] = None,
            min_lr: Optional[float] = None,
            ):
        if validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("AdamW beta values must be in [0, 1)")
        if grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")
        if optimizer_profile not in {"all-parameters", "nanogpt"}:
            raise ValueError("optimizer_profile must be 'all-parameters' or 'nanogpt'")
        if lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be 'constant' or 'cosine'")
        if lr_schedule == "cosine":
            if lr_decay_iters is None or lr_decay_iters <= warmup_iters:
                raise ValueError("cosine scheduling requires lr_decay_iters > warmup_iters")
            if min_lr is None or min_lr < 0:
                raise ValueError("cosine scheduling requires a non-negative min_lr")

        if optimizer_profile == "all-parameters":
            optimizer_groups = [{"params": list(self.parameters()), "weight_decay": weight_decay}]
        else:
            parameter_dict = {
                name: parameter
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            }
            optimizer_groups = [
                {
                    "params": [parameter for parameter in parameter_dict.values() if parameter.dim() >= 2],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [parameter for parameter in parameter_dict.values() if parameter.dim() < 2],
                    "weight_decay": 0.0,
                },
            ]

        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=lr,
            betas=(beta1, beta2),
        )

        def scheduled_lr(step: int) -> float:
            if lr_schedule == "constant":
                return lr
            if step < warmup_iters:
                return lr * (step + 1) / (warmup_iters + 1)
            if step > lr_decay_iters:
                return min_lr
            decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
            coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coefficient * (lr - min_lr)

        progress_bar = trange(epoch_num, desc="Training")

        losses = self.estimate_loss(manager=manager)
        loss_history = [{
            "step": 0,
            "train_loss": float(losses["train"].item()),
            "validation_loss": float(losses["validate"].item()),
            **manager.data_statistics(),
        }]
        loss_msg = f"loss_trn:{losses['train']:.4f}, loss_val:{losses['validate']:.4f}"
        progress_bar.set_postfix(loss=loss_msg)
        
        for bar_step in progress_bar:
            self.train()
            current_lr = scheduled_lr(bar_step)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr
            x, y, start_positions = self._unpack_training_batch(
                manager.get_batch("train", track_consumption=True)
            )
            logits, loss = self(x, y, start_positions=start_positions); loss: torch.Tensor
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()

            step = bar_step + 1
            if step % validation_interval == 0:
                losses = self.estimate_loss(manager=manager)
                loss_msg = f"loss_trn:{losses['train']:.4f}, loss_val:{losses['validate']:.4f}"
                progress_bar.set_postfix(loss=loss_msg)
                loss_history.append({
                    "step": step,
                    "train_loss": float(losses["train"].item()),
                    "validation_loss": float(losses["validate"].item()),
                    **manager.data_statistics(),
                })

        losses = self.estimate_loss(manager=manager)
        loss_msg = f"loss_trn:{losses['train']:.4f}, loss_val:{losses['validate']:.4f}"
        progress_bar.set_postfix(loss=loss_msg)
        if loss_history[-1]["step"] != epoch_num:
            loss_history.append({
                "step": epoch_num,
                "train_loss": float(losses["train"].item()),
                "validation_loss": float(losses["validate"].item()),
                **manager.data_statistics(),
            })
        return {
            **{split: float(loss.item()) for split, loss in losses.items()},
            "validation_interval": validation_interval,
            "optimizer": {
                "profile": optimizer_profile,
                "lr_schedule": lr_schedule,
                "lr": lr,
                "min_lr": min_lr,
                "warmup_iters": warmup_iters,
                "lr_decay_iters": lr_decay_iters,
                "weight_decay": weight_decay,
                "beta1": beta1,
                "beta2": beta2,
                "grad_clip": grad_clip,
            },
            "history": loss_history,
            "data_statistics": manager.data_statistics(),
        }
            

    def save_model(self, model_path = "iml_model.pth"):
        for param in self.parameters():
            param.data = param.data.to(torch.float32)
        parent_dir = os.path.dirname(model_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        torch.save(self.state_dict(), model_path, _use_new_zipfile_serialization=False)
        print(f"Model weights saved to {model_path}")


    def load_model(self, model_path="iml_model.pth"):
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.to(torch.float32)  # Ensure all parameters remain float32
        self.to(self.device).eval()
        print(f"Model weights loaded from {model_path}")


    def generate(
            self,
            batched_context: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 0,
            top_k: Optional[int] = None,
            syllable_num: int = 3,
            top_k_by_coordinate: Optional[Sequence[int]] = None,
            temperature_by_coordinate: Optional[Sequence[float]] = None,
            token_callback: Optional[Callable[[int], None]] = None,
            show_progress: bool = True,
            ):
        '''
        Understood
        '''
        if syllable_num <= 0:
            raise ValueError("syllable_num must be positive")
        if (
                self.ilm_input_embeddings
                or self.ilm_output_heads
                ) and syllable_num != self.syllable_num:
            raise ValueError("syllable_num must match the model syllable_num when role-aware maps are active")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if temperature < 0:
            raise ValueError("temperature must be zero or positive")
        if top_k_by_coordinate is not None and len(top_k_by_coordinate) != syllable_num:
            raise ValueError("top_k_by_coordinate must match syllable_num")
        if top_k_by_coordinate is not None and any(value <= 0 for value in top_k_by_coordinate):
            raise ValueError("top_k_by_coordinate values must be positive")
        if temperature_by_coordinate is not None and len(temperature_by_coordinate) != syllable_num:
            raise ValueError("temperature_by_coordinate must match syllable_num")
        if temperature_by_coordinate is not None and any(value < 0 for value in temperature_by_coordinate):
            raise ValueError("temperature_by_coordinate values must be zero or positive")
        if token_callback is not None and batched_context.shape[0] != 1:
            raise ValueError("token_callback only supports batch size 1")

        self.eval()
        batched_context_ = batched_context.to(self.device)
        progress_bar = trange(max_new_tokens, desc="Inference") if show_progress else range(max_new_tokens)
        # single_context should be (B, T) where T grows
        for _ in progress_bar:
            coordinate_index = batched_context_.shape[1] % syllable_num
            step_temperature = temperature
            if temperature_by_coordinate is not None:
                step_temperature = temperature_by_coordinate[coordinate_index]
            step_top_k = top_k
            if top_k_by_coordinate is not None:
                step_top_k = top_k_by_coordinate[coordinate_index]

            # give batched logits and loss
            batched_context_window = batched_context_[:, -self.block_size:]
            start_positions = None
            if (
                    self.ilm_input_embeddings
                    or self.ilm_output_heads
                    ):
                context_start = batched_context_.shape[1] - batched_context_window.shape[1]
                start_positions = torch.full(
                    (batched_context_window.shape[0],),
                    context_start % self.syllable_num,
                    dtype=torch.long,
                    device=self.device,
                )
            batched_logits, loss = self(
                batched_context_window,
                start_positions=start_positions,
            )
            # get the logits (prediction) for the last context token
            last_logits = batched_logits[:,-1,:]
   
            # compute associated probability distribution(s) (used in cross entropy)
            if step_temperature == 0:
                # Deterministic: choose the highest probability token
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                # Scale logits by the temperature if non-zero.
                #   - Lower temperatures (<1) make the softmax distribution sharper (more peaked),
                #     so the highest logit dominates.
                #   - Higher temperatures (>1) flatten the distribution, making the sampling more random.
                scaled_logits = last_logits / step_temperature

                if step_top_k is not None:
                    step_top_k = min(step_top_k, scaled_logits.shape[-1])
                    top_values, _ = torch.topk(scaled_logits, k=step_top_k)
                    threshold = top_values[:, -1].unsqueeze(-1)
                    scaled_logits = scaled_logits.masked_fill(scaled_logits < threshold, float("-inf"))

                # Compute the probability distribution over tokens.
                scaled_probs = F.softmax(scaled_logits, dim=-1)

                # Use torch.multinomial to sample the next token from the probability distribution.
                # IMPORTANT:
                # torch.multinomial is stochastic; it randomly draws an index based on the probabilities in 'scaled_probs'.
                # This means that even if one token has the highest probability, it might not always be selected.
                # If you need deterministic behavior (always selecting the token with the highest probability),
                # you would use torch.argmax instead.
                next_token = torch.multinomial(scaled_probs, num_samples=1)

            # extend the context to (B, T+1)
            batched_context_ = torch.cat((batched_context_, next_token), dim=1)
            if token_callback is not None:
                token_callback(int(next_token[0, 0].item()))

        return batched_context_
