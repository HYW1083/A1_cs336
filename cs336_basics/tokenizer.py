import os
from typing import BinaryIO
from multiprocessing import Queue, Process
import regex as re
from collections import Counter


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently. Chunk the files by <endoftext>.
    May return fewer chunks if the boundaries end up overlapping. For example, [0, 250, 250, 250, 400] -> [0, 250, 400]
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess 把文件读取位置移动到 initial_position 这个 byte 位置
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token) # find() return -1 if didn't find
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(text: str, special_tokens: list[str], drop_special_tokens: bool=True) -> Counter[tuple[bytes, ...]]:
    """
    Args:
        text: chunked text ready for pretokenization
        special_tokens: such as ["<|endoftext|>"]
        drop_special_tokens: drop special tokens or not
    Returns:
        pretokens(dict[tuple[bytes, ...], int]): a frequency map from byte-tokenized pretokens to counts, 
        for example, {(b"l", b"o", b"w"): 5}. Counter[tuple[bytes, ...]], here counter only receive one type of parameter, equal to dict[tuple[bytes, ...], int]
    """
    # Split by special tokens, could be ["<|endoftext|>", "text..."]
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True) # put "<|endoftext|><|endoftext|>" as a entire special token at first, if put "<|endoftext|>" at first, then this cannot be as a entire special token
    if not special_tokens_sorted:
        parts = [text]
    else:
        pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted) # 因为 special token 里有 |，而 | 在 regex 里表示“或者”，所以要转义。
        parts = re.split('(' + pattern + ')', text) # Use a capturing group ‘(‘ ’)‘ so re.split keeps the matched special tokens.
    # Delete those special tokens and use regular expression to pretokenize
    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
    for part in parts:
        if part in special_tokens:
            if not drop_special_tokens:
                token_bytes = part.encode('utf-8')
                pretoken = tuple(bytes([b]) for b in token_bytes)
                pretoken_counts[pretoken] += 1
        else:
            for match in re.finditer(PAT, part):
                token_str = match.group()        #从 Match 对象里取出这次正则匹配到的字符串，only return one string calling group()
                token_bytes = token_str.encode('utf-8') # remember to transfer to bytes.
                pretoken = tuple(bytes([b]) for b in token_bytes)
                pretoken_counts[pretoken] += 1
    return pretoken_counts

def worker(text: str, special_tokens: list[str], q: Queue):
    """Worker pretokenizes process for multiprocessing"""
    pretokens = pretokenize(text, special_tokens)
    q.put(pretokens)

def merge_word(word: tuple[bytes, ...],
               pair: tuple[bytes,bytes],
               ) -> tuple[bytes, ...]:
    merged = []
    i = 0

    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1

    return tuple(merged)

def get_pair_counts(word: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    """
    Args: 
        word(tuple[bytes, ...]): need to transfer to pair counter
    Returns:
        pair_counts(Counter[tuple[bytes, bytes]]): get paired with adjacent tokens, such as {(b'l', b'o'): 5, (b'o', b'w'): 5}
    """
    pair_counts = Counter()

    for pair in zip(word, word[1:]):
        pair_counts[pair] += 1

    return pair_counts

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Args:
        input_path (str | os.PathLike): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer that defines the maximum final vocabulary size
            (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens (list[str]): A list of strings to add to the vocabulary.
    Returns:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training.
    """
    special_tokens = [] if special_tokens is None else special_tokens
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must be at least 256 + len(special_tokens)")
    vocab = {}
    vocab = {x: bytes([x]) for x in range(0, 256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    chunk_list = []
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") # special tokens are at beginning of every chunk.
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_list.append(chunk)
        
    # Parallelizing pretokenization
    pretokens_list = []
    processes = []
    q = Queue()
    for chunk in chunk_list:
        p = Process(target=worker, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p) 

    pretokens_list = [q.get() for _ in processes] # Put all Counters together

    for p in processes:
        p.join()            # wait for the ending of every sub-process

    word_counts: Counter[tuple[bytes, ...]] = Counter()

    for chunk_counts in pretokens_list:
        word_counts.update(chunk_counts) # update() 合并相同的key

    pair_counts = Counter()
    pair_to_words: dict[tuple[bytes,bytes], set[tuple[bytes, ...]]] = {} 
    
    # pair_to_words[(b"l", b"o")] = {
      #(b"l", b"o", b"w"),
      #(b"l", b"o", b"w", b"e", b"r"),
    #}

    for word, count in word_counts.items():
        word_pair_counts = get_pair_counts(word) # here not consider the counts of word， so later the calculation must contain pair_count * count

        for pair, pair_count in word_pair_counts.items():
            pair_counts[pair] += count * pair_count  # calculate the numbers

            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)  # document the position of pair (which word contains this pair)

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key = lambda pair: (pair_counts[pair], pair))

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected_words = list(pair_to_words.get(best_pair, set()))

        new_words_to_add: Counter[tuple[bytes, ...]] = Counter()

        for old_word

    # Calculate from word_counts to pair_counts each time
    # merges: list[tuple[bytes, bytes]] = []

    # while len(vocab) < vocab_size:
    #     pair_counts: Counter[tuple[bytes,bytes]] = Counter()

    #     for word, count in word_counts.items():
    #         for pair in zip(word, word[1:]):        # word跟word[1:]逐个配对
    #             pair_counts[pair] += count          # pair_counts[pair] is the frequency(value) of this pair(key)

    #     if not pair_counts:
    #         break

    #     best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))  # Prefer lexicographically greater pair
    #     # Example: max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]) = ('BA', 'A')

    #     merges.append(best_pair)
    #     vocab[len(vocab)] = best_pair[0] + best_pair[1]
        
    #     new_word_counts: Counter[tuple[bytes, ...]] = Counter()

    #     for word, count in word_counts.items():
    #         new_word = merge_word(word, best_pair)
    #         new_word_counts[new_word] += count

    #     word_counts = new_word_counts

    # return vocab, merges


class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):

