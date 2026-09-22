import json
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer, train_bpe

def main():
    # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10000
    # special_tokens = ["<|endoftext|>"]
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    # output_dir = Path("artifacts/tokenizer_ts")
    output_dir = Path("artifacts/tokenizer_owt")
    output_dir.mkdir(parents = True, exist_ok = True)

    # 训练
    start_time = time.perf_counter()     # performance counter 性能计数器

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    elapsed = time.perf_counter() - start_time

    print(f"Total BPE training time: {elapsed:.2f} s")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # 保存词表和合并规则：需要先把字节串转化成JSON支持的整数列表，因为JSON不支持python的bytes类型
    # vocab = {                     vocab_data = {
    # 0: b"\x00",                   0: [0],
    # 108: b"l",      转化为：       108: [108]，
    # 256: b"lo",                   256: [108, 111]
    # }                             }

    vocab_data = {
        token_id: list(token_bytes)
        for token_id, token_bytes in vocab.items()
        }
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w", encoding = "utf-8") as f:
        json.dump(vocab_data, f)

    merges_data = [
        [list(left), list(right)]
        for left, right in merges]
    merges_path = output_dir / "merges.json"
    with open(merges_path, "w", encoding = "utf-8") as f:
        json.dump(merges_data, f)

    # 检查最长的普通token
    special_bytes = {
        token.encode("utf-8") for token in special_tokens
    }
    longest_token = max((token for token in vocab.values() if token not in special_bytes), key = len)
    print(f"Longest non-special token: {longest_token!r}")
    print(f"Token length: {len(longest_token)} bytes")

if __name__ == "__main__":
    main()