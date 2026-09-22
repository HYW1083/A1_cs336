from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
# .parent: experiments文件夹
# .parent.parent: A1_cs336项目根目录
SPECIAL_TOKEN = "<|endoftext|>"


def read_first_documents(
    input_path: Path,
    num_documents: int = 10,
) -> list[str]:
    """逐行读取文件，提取前 num_documents 篇非空文档。"""
    documents = []
    buffer = ""

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            buffer += line

            # 一行中可能出现多个文档结束标记
            while SPECIAL_TOKEN in buffer:
                document, buffer = buffer.split(SPECIAL_TOKEN, 1)

                # strip() 只用于判断是否为空，不修改保存的正文
                if document.strip():
                    documents.append(document)

                if len(documents) == num_documents:
                    return documents

    raise ValueError(f"Found fewer than {num_documents} non-empty documents.")


def evaluate_compression(
    dataset_name: str,
    input_path: Path,
    tokenizer_dir: Path,
) -> float:
    """加载 tokenizer，计算前 10 篇文档的整体 bytes/token。"""
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_dir / "vocab.json",
        merges_filepath=tokenizer_dir / "merges.json",
        special_tokens=[SPECIAL_TOKEN],
    )

    documents = read_first_documents(
        input_path=input_path,
        num_documents=10,
    )

    total_bytes = 0
    total_tokens = 0

    print(f"\nDataset: {dataset_name}")
    print(f"Tokenizer: {tokenizer_dir.name}")
    print(f"{'Document':>10} {'Bytes':>12} {'Tokens':>12} {'Bytes/token':>14}")

    for index, document in enumerate(documents, start=1):
        # 原始文本的 UTF-8 字节数
        num_bytes = len(document.encode("utf-8"))

        # 编码后的 token 数
        token_ids = tokenizer.encode(document)
        num_tokens = len(token_ids)

        # 检查编码、解码能够还原原始文本
        if tokenizer.decode(token_ids) != document:
            raise ValueError(
                f"Round-trip check failed for {dataset_name}, "
                f"document {index}."
            )

        ratio = num_bytes / num_tokens

        total_bytes += num_bytes
        total_tokens += num_tokens

        print(
            f"{index:>10} "
            f"{num_bytes:>12,} "
            f"{num_tokens:>12,} "
            f"{ratio:>14.4f}"
        )

    # 整体压缩率：总字节数 / 总 token 数
    compression_ratio = total_bytes / total_tokens

    print(f"Total documents: {len(documents)}")
    print(f"Total UTF-8 bytes: {total_bytes:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Compression ratio: {compression_ratio:.4f} bytes/token")

    return compression_ratio


def main():
    ts_ratio = evaluate_compression(
        dataset_name="TinyStories",
        input_path=ROOT / "data/TinyStoriesV2-GPT4-valid.txt",
        tokenizer_dir=ROOT / "artifacts/tokenizer_ts",
    )

    owt_ratio = evaluate_compression(
        dataset_name="OpenWebText",
        input_path=ROOT / "data/owt_valid.txt",
        tokenizer_dir=ROOT / "artifacts/tokenizer_owt",
    )

    owt_with_ts_ratio = evaluate_compression(
        dataset_name="OpenWebText with TinyStories tokenizer",
        input_path=ROOT / "data/owt_valid.txt",
        tokenizer_dir=ROOT / "artifacts/tokenizer_ts",
    )

    print("\nSummary")
    print(f"TinyStories: {ts_ratio:.4f} bytes/token")
    print(f"OpenWebText: {owt_ratio:.4f} bytes/token")
    print(f"TinyStories tokenizer on OWT: {owt_with_ts_ratio:.4f} bytes/token")

if __name__ == "__main__":
    main()