import time
from pathlib import Path
from statistics import median

from cs336_basics.tokenizer import Tokenizer
from tokenizer_exp_a import read_first_documents


ROOT = Path(__file__).resolve().parent.parent
SPECIAL_TOKEN = "<|endoftext|>"


def benchmark(
    dataset_name: str,
    input_path: Path,
    tokenizer_dir: Path,
    num_documents: int = 100,
    repeats: int = 3,
):
    # 加载 tokenizer，不计入编码耗时
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_dir / "vocab.json",
        merges_filepath=tokenizer_dir / "merges.json",
        special_tokens=[SPECIAL_TOKEN],
    )

    # 读取文档，不计入编码耗时
    documents = read_first_documents(
        input_path=input_path,
        num_documents=num_documents,
    )

    # 恢复文档之间的分隔标记
    text = SPECIAL_TOKEN.join(documents) # 加入special token作为分隔符, 其中join用法：" ".join(["hello", "world"])  # "hello world"
    num_bytes = len(text.encode("utf-8"))

    # 预热一次，减少首次调用的影响
    warmup_ids = tokenizer.encode(text)
    del warmup_ids

    durations = []
    num_tokens = 0

    for _ in range(repeats):
        start_time = time.perf_counter()

        ids = tokenizer.encode(text)

        elapsed = time.perf_counter() - start_time
        durations.append(elapsed)

        num_tokens = len(ids)
        del ids

    # 使用多次测量的中位数
    elapsed = median(durations)
    bytes_per_second = num_bytes / elapsed

    # 按十进制单位计算：1 GB = 10^9 bytes
    pile_bytes = 825 * 10**9
    estimated_seconds = pile_bytes / bytes_per_second

    print(f"\nDataset: {dataset_name}")
    print(f"Number of documents: {len(documents)}")
    print(f"Input size: {num_bytes:,} bytes")
    print(f"Number of tokens: {num_tokens:,}")
    print(f"Encoding times: {[round(t, 3) for t in durations]} s")
    print(f"Median encoding time: {elapsed:.3f} s")
    print(f"Throughput: {bytes_per_second:,.2f} bytes/s")
    print(f"Throughput: {bytes_per_second / 10**6:.3f} MB/s")
    print(
        f"Estimated time to tokenize 825 GB: "
        f"{estimated_seconds / 3600:.2f} hours "
        f"({estimated_seconds / 86400:.2f} days)"
    )


def main():
    benchmark(
        dataset_name="TinyStories",
        input_path=ROOT / "data/TinyStoriesV2-GPT4-valid.txt",
        tokenizer_dir=ROOT / "artifacts/tokenizer_ts",
    )

    benchmark(
        dataset_name="OpenWebText",
        input_path=ROOT / "data/owt_valid.txt",
        tokenizer_dir=ROOT / "artifacts/tokenizer_owt",
    )


if __name__ == "__main__":
    main()