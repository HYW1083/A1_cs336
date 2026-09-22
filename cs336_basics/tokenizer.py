import os
import time
from typing import BinaryIO, Iterable
from multiprocessing import Queue, Process
import regex as re # regular expression “正则表达式”，本质上是一种描述文本模式的字符串。例如r"\d+" 表示一个或多个数字；r"[a-z]+"表示一个或多个小写英文字母
from collections import Counter
import json


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
        parts = re.split('(' + pattern + ')', text) # Use a capturing group ‘(‘ ’)‘ so re.split keeps the matched special tokens. 注意这里的pattern是sp
        # "some text<|endoftext|>hello world" -> ["some text", "<|endoftext|>","hello world"]

    # Delete those special tokens and use regular expression to pretokenize
    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
    for part in parts:
        if part in special_tokens:
            continue
        else:
            for match in re.finditer(PAT, part):
                token_str = match.group()        #从 Match 对象里取出这次正则匹配到的字符串，例如for match in re.finditer(PAT, "some text"): match.group() == some
                token_bytes = token_str.encode('utf-8') # remember to transfer to bytes. b'some'
                pretoken = tuple(bytes([b]) for b in token_bytes) # (b's', b'o', b'm', b'e')
                pretoken_counts[pretoken] += 1
    return pretoken_counts

def worker(text: str, special_tokens: list[str], q: Queue):
    """Worker pretokenizes process for multiprocessing"""
    pretokens = pretokenize(text, special_tokens)
    q.put(pretokens)

def merge_word(word: tuple[bytes, ...],
               pair: tuple[bytes,bytes],
               ) -> tuple[bytes, ...]:
    """
    Merge every non-overlapping occurrence of an adjacent token pair in a pre-token.

    Args:
        word: The current byte-token representation of a pre-token.
            Each element is either a single byte or a byte sequence produced
            by an earlier BPE merge. word = (b"l", b"o", b"w")
        pair: The adjacent pair of byte tokens to merge. The first element
            must be immediately followed by the second element to be merged.pair = (b"o", b"w")

    Returns:
        A new tuple in which every non-overlapping occurrence of `pair`
        has been replaced by the concatenation of the two byte tokens.
        The input `word` is not modified. (b"l", b"ow")
    """

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

def get_pair(word: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    """
    Args: 
        word(tuple[bytes, ...]): need to transfer to pair counter
    Returns:
        pair_counts(Counter[tuple[bytes, bytes]]): get paired with adjacent tokens, such as {(b'l', b'o'): 1, (b'o', b'w'): 1}
    """
    pair_counts = Counter()
    # get_pair() 只负责统计 pair 在“当前一个 word 内”出现了多少次，它并不知道这个 word 在整个语料中出现了多少次。
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
    vocab = {}       # 字典 dict
    vocab = {x: bytes([x]) for x in range(0, 256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    t0 = time.perf_counter()
    chunk_list = []
    with open(input_path, "rb") as f: # rb: read binary. 以二进制只读模式打开文件；with 的作用是在代码执行结束后自动关闭文件。
        num_processes = 16
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)           # 把文件当前的读取位置移动到 start
            chunk = f.read(end - start).decode("utf-8", errors="ignore") # special tokens are at beginning of every chunk.
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_list.append(chunk)
    t1 = time.perf_counter()  
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
    t2 = time.perf_counter()
    """
    Optimized method:
    Cache pair_counts and pair_to_words so each merge only updates pretokens that contain best_pair. 
    For affected words, remove old pair contributions, merge the word, and add the new pair contributions back. 
    This avoids recomputing pair_counts from all pretokens after every merge.
    """

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes,bytes], set[tuple[bytes, ...]]] = {} 
    
    # word_counts = Counter({
    #     (b"l", b"o", b"w"): 5,
    #     (b"l", b"o", b"w", b"e", b"r"): 2,
    # })

    # pair_counts = Counter({
    #     (b"l", b"o"): 7,
    #     (b"o", b"w"): 7,
    #     (b"w", b"e"): 2,
    #     (b"e", b"r"): 2,
    # })

    # pair_to_words = {
    #     (b"l", b"o"): {
    #         (b"l", b"o", b"w"),
    #         (b"l", b"o", b"w", b"e", b"r"),
    #     },
    # }

    for word, count in word_counts.items():
        word_pair_counts = get_pair(word) # here not consider the counts of word， so later the calculation must contain pair_count * count

        for pair, pair_count in word_pair_counts.items():
            pair_counts[pair] += count * pair_count  # 整个语料中的个数count，乘以当前pair在一个word中出现的次数pair_count 
            # 例如 word_counts = Counter({ (b"l", b"o", b"w"): 5, (b"l", b"o", b"w", b"e", b"r"): 2,}) 的(b"l", b"o") = 5*1 + 2*1

            if pair not in pair_to_words:
                pair_to_words[pair] = set() #第一次遇到这个pair先设置空集合初始化。set()没有会KeyError
            pair_to_words[pair].add(word)  # document the position of pair (which word contains this pair)
    t3 = time.perf_counter()
    merges: list[tuple[bytes, bytes]] = [] # 注意merges 是列表，不是字典

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key = lambda pair: (pair_counts[pair], pair)) # max(iterable, key = 比较函数)， key： 按照什么标准比较。
        # max() 会：1. 遍历 iterable 中的每个元素；2. 把每个元素传给 key 函数；3. 比较 key 函数的返回值；4. 返回原始元素，而不是 key 的返回值。
        # 在这里会先按照pair_counts[pair]来进行比较，之后再按照pair的字典序比较
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected_words = list(pair_to_words.get(best_pair, set()))

        new_words_to_add: Counter[tuple[bytes, ...]] = Counter()

        for old_word in affected_words:
            if old_word in word_counts:
                count = word_counts[old_word]
                del word_counts[old_word]    # or combine them as: count = word_counts.pop(old_word, 0) if count == 0: continue
            else:
                continue

            old_pair_counts= get_pair(old_word)
            # Remove this old_word's contribution from pair_counts and pair_to_words.
            for pair, pair_count in old_pair_counts.items():  # 遍历旧单词中的每个pair.
                pair_counts[pair] -= pair_count * count

                if pair_counts[pair] <= 0:     # 当完全没有这个pair时，删掉这个key, 若不删会保留这个旧key等于0的情况
                    del pair_counts[pair]

                if pair in pair_to_words: 
                    pair_to_words[pair].discard(old_word)      # 删掉pair_to_words中的旧word

                    if not pair_to_words[pair]:
                        del pair_to_words[pair]

            new_word = merge_word(old_word, best_pair)
            new_words_to_add[new_word] += count

        for new_word, count in new_words_to_add.items():
            word_counts[new_word] += count     # 自动创建

            new_pair_counts = get_pair(new_word)

            for pair, pair_count in new_pair_counts.items():
                pair_counts[pair] += pair_count * count

                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(new_word)
    t4 = time.perf_counter()
    print(f"File reading and chunking: {t1 - t0:.2f} s")
    print(f"Parallel pre-tokenization and aggregation: {t2 - t1:.2f} s")
    print(f"Pair count and index initialization: {t3 - t2:.2f} s")
    print(f"BPE merging: {t4 - t3:.2f} s")
    return vocab, merges    

    # """ Original method: Calculate from word_counts to pair_counts each time"""

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
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str]| None = None):
        self.vocab = vocab
        self.merges = merges
        # 建立反向词表，在encode阶段使用: bytes -> token ID
        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
            }
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
            }
        self.special_tokens = []
        if special_tokens is not None:
            for token in special_tokens:
                if not token:
                    raise ValueError("Special tokens must not be empty")
                if token not in self.special_tokens:
                    self.special_tokens.append(token)

                token_bytes = token.encode("utf-8")

                if token_bytes not in self.token_to_id:
                    new_id = max(self.vocab, default=-1) + 1

                    self.vocab[new_id] = token_bytes
                    self.token_to_id[token_bytes] = new_id

    def _encode_pretoken(self, text: str) -> list[int]:
        encoded_text = text.encode("utf-8")
        byte_tokens = []
        for value in encoded_text:
            byte_tokens.append(bytes([value]))
        word = tuple(byte_tokens)        #(b"l", b"o", b"w")
        #不断寻找可以合并的相邻pair
        while len(word) > 1:
            best_pair = None
            best_rank = float("inf")

            for pair in zip(word, word[1:]):
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]

                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            word = merge_word(word, best_pair)
            
        return [self.token_to_id[token] for token in word]

    def encode(self, text: str) -> list[int]:
        #先按照special_token切分，保留特殊token本身
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(token) for token in sorted_tokens)
            parts = re.split("(" + special_pattern + ")", text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if part in self.special_tokens:
                token_bytes = part.encode("utf-8")
                ids.append(self.token_to_id[token_bytes])

            else:
                for match in re.finditer(PAT, part):
                    pretoken = match.group()  #转化为多个字符片段例如'some'，' text'， ' that'， ' i'， "'ll"， ' pre'， '-'， 'tokenize'

                    pretoken_ids = self._encode_pretoken(pretoken)
                    ids.extend(pretoken_ids)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            ids = self.encode(text)

            for token_id in ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        byte_parts = []

        for token_id in ids:
            token_bytes = self.vocab[token_id]
            byte_parts.append(token_bytes)

        all_bytes = b"".join(byte_parts)

        return all_bytes.decode("utf-8", errors="replace")
    
    @classmethod
    def from_files(cls, 
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None):
        # 读取JSON数据 因为某些 token 只包含汉字的一部分 UTF-8 字节，不能独立解码为正常字符串 
        # # 内存里的词表 {256: b"lo"}
        # # 保存成 JSON {"256": [108, 111]}
        with open(vocab_filepath, "r", encoding="utf-8") as f: 
            vocab_data = json.load(f)

        with open(merges_filepath, "r", encoding = "utf-8") as f:
            merges_data = json.load(f)
        # 字符串转回int
        vocab = {}

        for token_id, byte_value in vocab_data.items():
            vocab[int(token_id)] = bytes(byte_value)
        # Json保存的整数转化为merges中的合并的字节对
        #merges_data = [
        #     [[108], [111]],
        #     [[108, 111], [119]],
        # ]
        merges = []
        for left, right in merges_data:
            pair = (bytes(left), bytes(right))
            merges.append(pair)

        return cls(vocab, merges, special_tokens)




