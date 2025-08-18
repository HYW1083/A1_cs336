import os
import re
from collections import Counter
from typing import Iterable, Iterator

# Regular expression for GPT-2 tokenization (approximation without regex module)
TOKEN_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\\W\\d_]+| ?\\d+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+",
    re.UNICODE,
)


def _tokenize(text: str, special_tokens: list[str] | None) -> list[str]:
    """Split text into tokens, keeping special tokens intact."""
    if not special_tokens:
        return TOKEN_PATTERN.findall(text)
    tokens: list[str] = []
    i = 0
    last = 0
    specials = set(special_tokens)
    length = len(text)
    while i < length:
        match = None
        for s in specials:
            if text.startswith(s, i):
                match = s
                break
        if match:
            if last < i:
                tokens.extend(TOKEN_PATTERN.findall(text[last:i]))
            tokens.append(match)
            i += len(match)
            last = i
        else:
            i += 1
    if last < length:
        tokens.extend(TOKEN_PATTERN.findall(text[last:]))
    return tokens


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a simple byte pair encoder on the given corpus."""
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_strings = _tokenize(text, special_tokens or [])

    token_freq = Counter(token_strings)

    # Represent each token as a tuple of byte strings
    word_freq: dict[tuple[bytes, ...], int] = {}
    specials = set(special_tokens or [])
    for token, freq in token_freq.items():
        if token in specials:
            word = (token.encode("utf-8"),)
        else:
            word = tuple(bytes([b]) for b in token.encode("utf-8"))
        word_freq[word] = freq

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) + len(specials) < vocab_size:
        # Count adjacent pairs
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += freq
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)
        new_symbol = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_symbol

        # Replace occurrences of best_pair in all words
        new_word_freq: dict[tuple[bytes, ...], int] = {}
        first, second = best_pair
        for word, freq in word_freq.items():
            i = 0
            new_word: list[bytes] = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(new_symbol)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freq[tuple(new_word)] = freq
        word_freq = new_word_freq

    # Append special tokens to vocab
    for token in specials:
        vocab[len(vocab)] = token.encode("utf-8")

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or []
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        word = [bytes([b]) for b in token_bytes]
        if not word:
            return []
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            if not pairs:
                break
            ranked = [self.bpe_ranks.get(p, float("inf")) for p in pairs]
            min_rank = min(ranked)
            if min_rank == float("inf"):
                break
            idx = ranked.index(min_rank)
            first, second = word[idx], word[idx + 1]
            word = (
                word[:idx]
                + [first + second]
                + word[idx + 2 :]
            )
        return word

    def _encode_generator(self, text: str) -> Iterator[int]:
        tokens = _tokenize(text, self.special_tokens)
        specials = {s: self.byte_to_id[s.encode("utf-8")] for s in self.special_tokens}
        for token in tokens:
            if token in specials:
                yield specials[token]
            else:
                token_bytes = token.encode("utf-8")
                for piece in self._bpe(token_bytes):
                    yield self.byte_to_id[piece]

    def encode(self, text: str) -> list[int]:
        return list(self._encode_generator(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self._encode_generator(chunk)

    def decode(self, ids: Iterable[int]) -> str:
        output = b"".join(self.vocab[i] for i in ids)
        return output.decode("utf-8", errors="replace")


__all__ = ["train_bpe", "Tokenizer"]
