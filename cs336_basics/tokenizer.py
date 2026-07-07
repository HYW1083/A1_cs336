import torch
import os
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
)
    """
    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A positive integer that defines the maximum final vocabulary size
            (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: A list of strings to add to the vocabulary. During training, treat them as hard boundaries 
            that prevent merges across their spans, but do not include them when computing merge statistics.
    Returns:
        
    """