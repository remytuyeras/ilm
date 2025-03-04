"""
This module provides utilities for generating a hierarchical tokenizer and detokenizer from a source text file.
It reads the file, extracts tokens using regular expressions, computes relative positional weights across different
levels (e.g., sentence, paragraph), classifies tokens into bins, and finally produces a mapping from tokens to codes 
and vice versa. The resulting mapping can be saved or loaded from a JSON file.
"""

import os
import re
import json
from collections import OrderedDict
from typing import Tuple, Optional, Dict, List, Callable, Any

import numpy as np
from tqdm import tqdm  # progress bar


def save_dictionary(mapping: Dict[str, Any], filename: str = "dictionary.json") -> None:
    """
    Save a dictionary to a JSON file with pretty-print formatting.

    Args:
        mapping: Dictionary to be saved.
        filename: Target filename.
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=4)


def load_dictionary(filename: str = "dictionary.json") -> Dict[str, Any]:
    """
    Load a dictionary from a JSON file.

    Args:
        filename: The JSON file to load from.

    Returns:
        The loaded dictionary.
    """
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def force_json_extension(file_path: str) -> str:
    """
    Ensure the given file path has a .json extension.

    Args:
        file_path: Original file path.

    Returns:
        File path with .json extension.
    """
    base, _ = os.path.splitext(file_path)
    return f"{base}.json"

def find_tokens(text: str) -> List[str]:
    """
    Extract tokens from the input text using a regular expression pattern.
    The pattern captures LaTeX math brackets, words (with or without leading spaces),
    and standalone punctuation.

    Args:
        text: Input text.

    Returns:
        List of extracted tokens.
    """
    pattern = (
    r'(\[|\]|\$\$|\$|\\\[|\\\]|\\\(|\\\)|\\|'       # match specific bracket tokens or a literal backslash
    r'[\n\r\t]|'                                    # match any actual newline, carriage return, or tab
    r'[ ]+[ ]|'                                     # match two or more spaces
    r'[ \f\v]+[A-Za-z0-9]+|'                        # match leading whitespace (except newline) plus a word
    r'[A-Za-z0-9]+|'                                # match words
    r'[.,:;()!?+\-_*&^%#={}\'\"])'                  # match punctuation or quotes
    )
    return re.findall(pattern, text)


def compute_token_weights(source_file: str, level: int) -> Dict[str, float]:
    """
    Compute the average relative position of each token from a source file.
    The relative position is computed per token occurrence in the split sentences.
    The level parameter chooses the splitting regex: 
      0 - using comma or dot as boundary,
      1 - using dot as boundary,
      2 - using newline as boundary.

    Args:
        source_file: Path to the source text file.
        level: Level of text segmentation (0, 1, or 2).

    Returns:
        Dictionary mapping tokens to their average relative position weight.
    """
    token_weights: Dict[str, List[float]] = {}
    # Define regular expressions for different levels of segmentation.
    split_patterns = [
        r'(?<=\,|\.|\n|\r)+',  # Level 0: split on comma, dot or newline
        r'(?<=\.|\n|\r)+',     # Level 1: split on dot or newline
        r'(?<=\n|\r)+'      # Level 2: split on newline
    ]
    pattern = split_patterns[level]

    with open(source_file, "r", encoding="utf-8") as file:
        for line in file:
            # Strip leading/trailing whitespace and split the line based on the chosen pattern.
            segments = re.split(pattern, line)
            for segment in segments:
                tokens = find_tokens(segment)
                token_count = len(tokens)
                if token_count == 0:
                    continue
                # Use enumeration to correctly capture the index for each token occurrence.
                for idx, token in enumerate(tokens):
                    token_weights.setdefault(token, []).append(idx / token_count)
    # Average the weights for each token.
    averaged_weights = {token: float(np.mean(weights)) for token, weights in token_weights.items()}
    # Ensure newline characters are included, as they were removed during line processing.
    if "\n" in averaged_weights:
        print("Error", averaged_weights["\n"])
    averaged_weights.setdefault("\n", 1)
    averaged_weights.setdefault("\r", 1)
    return averaged_weights


def classify_tokens(weighted_tokens: Dict[str, float]) -> Dict[str, int]:
    """
    Classify tokens into bins based on their weight.
    The tokens are sorted by weight, and binned into up to 64 bins.

    Args:
        weighted_tokens: A dictionary mapping tokens to their weights.

    Returns:
        A dictionary mapping each token to its bin index.
    """
    sorted_tokens = sorted(weighted_tokens.items(), key=lambda x: x[1])
    num_bins = min(64, len(sorted_tokens))
    bin_size = max(len(sorted_tokens) // num_bins, 1)
    token_bins: Dict[str, int] = {}
    for i, (token, _) in enumerate(sorted_tokens):
        bin_index = min(i // bin_size, num_bins - 1)
        token_bins[token] = bin_index
    return token_bins


def weight_classified_tokens(
    classified_tokens: Dict[str, int],
    weighted_tokens: Optional[Dict[str, float]] = None
) -> Dict[int, Dict[str, Optional[float]]]:
    """
    Organize classified tokens into bins, optionally including their weights.

    Args:
        classified_tokens: A dictionary mapping tokens to their bin indices.
        weighted_tokens: Optional dictionary mapping tokens to their weights.

    Returns:
        Dictionary where keys are bin indices and values are dictionaries mapping tokens 
        in that bin to their weights (or None if not provided). The bins are sorted by key.
    """
    bins: Dict[int, Dict[str, Optional[float]]] = {}
    for token, bin_index in classified_tokens.items():
        bins.setdefault(bin_index, {})
        bins[bin_index][token] = weighted_tokens[token] if weighted_tokens is not None else None
    # Sort bins by their keys.
    return dict(OrderedDict(sorted(bins.items())))


def generate_tokenizer(mapping: Dict[str, str]) -> Callable[[str], List[Optional[str]]]:
    """
    Generate a tokenizer function using a direct mapping from tokens to codes.

    Args:
        mapping: A dictionary mapping tokens to their code strings.

    Returns:
        A function that tokenizes an input string into a list of code strings.
    """
    def tokenizer(text: str) -> List[Optional[str]]:
        tokens = find_tokens(text)
        return [mapping.get(token, None) for token in tokens]
    return tokenizer


def generate_detokenizer(reverse_mapping: Dict[str, str]) -> Callable[[List[str]], List[Optional[str]]]:
    """
    Generate a detokenizer function using a reverse mapping from codes to tokens.

    Args:
        reverse_mapping: A dictionary mapping code strings back to tokens.

    Returns:
        A function that converts a list of code strings back to their corresponding tokens.
    """
    def detokenizer(code_sequence: List[str]) -> List[Optional[str]]:
        return [reverse_mapping.get(code, None) for code in code_sequence]
    return detokenizer


def create_tokenizer(
    source_file: str, 
    target_file: Optional[str] = None
) -> Tuple[Callable[[str], List[Optional[str]]], Callable[[List[str]], List[Optional[str]]]]:
    """
    Create a hierarchical tokenizer and detokenizer from the source file.
    The process involves computing token weights at three segmentation levels,
    classifying tokens into bins, and building a mapping where each token is 
    assigned a composite code in the form "level0:level1:level2".

    Progress bars (via tqdm) are added to track the progress of the nested binning loops.

    Args:
        source_file: Path to the source text file.
        target_file: Optional path to save the generated tokenizer mapping as a JSON file.

    Returns:
        A tuple containing:
          - A tokenizer function that converts strings to lists of token codes.
          - A detokenizer function that converts lists of token codes back to strings.
    """
    # Compute token weights at three segmentation levels.
    weights_level0 = compute_token_weights(source_file, 0)
    weights_level1 = compute_token_weights(source_file, 1)
    weights_level2 = compute_token_weights(source_file, 2)
    weights = [weights_level0, weights_level1, weights_level2]

    # First-level binning using level 0 weights and level 1 weights for weighting bins.
    level0_bins = weight_classified_tokens(classify_tokens(weights[0]), weights[1])

    # Second-level binning: refine each level0 bin using level 2 weights.
    for bin_idx, tokens_dict in tqdm(level0_bins.items(), desc="Processing Level 1 bins"):
        refined_bins = weight_classified_tokens(classify_tokens(tokens_dict), weights[2])
        level0_bins[bin_idx] = refined_bins

    # Third-level binning: further refine each second-level bin.
    for level0_index, level1_bins in tqdm(level0_bins.items(), desc="Processing Level 2 bins"):
        for level1_index, token_subset in tqdm(level1_bins.items(), desc="Processing Level 3 bins", leave=False):
            further_refined = weight_classified_tokens(classify_tokens(token_subset))
            level1_bins[level1_index] = further_refined

    # Build the final mapping from tokens to composite codes and vice versa.
    token_mapping: Dict[str, Dict[str, Any]] = {"direct": {}, "reverse": {}}
    for level0_index, level1_bins in tqdm(level0_bins.items(), desc="Generating mapping: Level 1"):
        for level1_index, level2_bins in tqdm(level1_bins.items(), desc="Generating mapping: Level 2", leave=False):
            for level2_index, tokens in tqdm(level2_bins.items(), desc="Generating mapping: Level 3", leave=False):
                # Get the first (and only) token from the dictionary.
                token = next(iter(tokens.keys()))
                composite_code = f"{level0_index}:{level1_index}:{level2_index}"
                token_mapping["direct"].setdefault(token, composite_code)
                token_mapping["reverse"].setdefault(composite_code, token)

    # Save the mapping if a target file is provided.
    if target_file is not None:
        save_dictionary(token_mapping, force_json_extension(target_file))

    return generate_tokenizer(token_mapping["direct"]), generate_detokenizer(token_mapping["reverse"])


def load_tokenizer(source_file: str) -> Tuple[Callable[[str], List[Optional[str]]], Callable[[List[str]], List[Optional[str]]]]:
    """
    Load the tokenizer and detokenizer from a JSON file generated by create_tokenizer().

    Args:
        source_file: Path to the JSON file containing the tokenizer mapping.

    Returns:
        A tuple containing:
          - A tokenizer function.
          - A detokenizer function.
    """
    token_mapping = load_dictionary(source_file)
    return generate_tokenizer(token_mapping["direct"]), generate_detokenizer(token_mapping["reverse"])
