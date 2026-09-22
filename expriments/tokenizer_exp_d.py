import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parent.parent
SPECIAL_TOKEN = "<|endoftext|>"

    
def iter_documents(input_path: Path):
    """逐篇读取文档，保留每篇结尾的特殊 token。"""
    buffer = ""

    with open(input_path, encoding="utf-8", newline="") as f:
        for line in f:
            buffer += line

            while True:
                position = buffer.find(SPECIAL_TOKEN)

                if position == -1: # 没找到退出while循环
                    break

                # 切分位置位于特殊 token 之后
                end = position + len(SPECIAL_TOKEN)

                document = buffer[:end]
                buffer = buffer[end:]

                yield document
    # 保留文件末尾剩余的文本，包括空白
    if buffer:
        yield buffer


def encode_file(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
):
    """逐篇编码并写入 uint16 二进制文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 在转换成 uint16 之前检查 ID 范围，避免溢出
    max_uint16 = np.iinfo(np.uint16).max

    if min(tokenizer.vocab) < 0 or max(tokenizer.vocab) > max_uint16:
        raise ValueError("Vocabulary IDs cannot be represented as uint16.")

    total_tokens = 0
    total_documents = 0
    start_time = time.perf_counter()

    # wb 表示二进制写入，已有文件会被覆盖
    with open(output_path, "wb") as output_file:
        for document in iter_documents(input_path):
            ids = tokenizer.encode(document)

            # 固定为小端、2 字节无符号整数
            token_array = np.asarray(ids, dtype="<u2")
            token_array.tofile(output_file)

            total_tokens += len(ids)
            total_documents += 1

            if total_documents % 10000 == 0:
                print(
                    f"{input_path.name}: "
                    f"{total_documents:,} chunks encoded, "
                    f"{total_tokens:,} tokens",
                    flush=True,
                )

    elapsed = time.perf_counter() - start_time
    output_bytes = output_path.stat().st_size

    # 每个 uint16 应占 2 字节
    assert output_bytes == total_tokens * 2

    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Output size: {output_bytes / 10**6:.2f} MB")
    print(f"Total elapsed time: {elapsed:.2f} s")


def main():
    output_dir = ROOT / "data/tokenized"

    datasets = [
        (
            "tokenizer_ts",
            [
                ("TinyStoriesV2-GPT4-valid.txt", "tinystories_valid.bin"),
                ("TinyStoriesV2-GPT4-train.txt", "tinystories_train.bin"),
            ],
        ),
        (
            "tokenizer_owt",
            [
                ("owt_valid.txt", "owt_valid.bin"),
                ("owt_train.txt", "owt_train.bin"),
            ],
        ),
    ]

    for tokenizer_name, files in datasets:
        tokenizer_dir = ROOT / "artifacts" / tokenizer_name

        tokenizer = Tokenizer.from_files(
            vocab_filepath=tokenizer_dir / "vocab.json",
            merges_filepath=tokenizer_dir / "merges.json",
            special_tokens=[SPECIAL_TOKEN],
        )

        for input_name, output_name in files:
            encode_file(
                tokenizer=tokenizer,
                input_path=ROOT / "data" / input_name,
                output_path=output_dir / output_name,
            )


if __name__ == "__main__":
    main()