from pathlib import Path
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent
PROMPT = "Once upon a time, there was a little girl"

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    model.eval()
    device = next(model.parameters()).device

    # 将提示文本编码成 token IDs. ids: list[int]
    ids = tokenizer.encode(prompt)

    if not ids:
        raise ValueError("prompt must contain at least one token.")

    eos_id = tokenizer.token_to_id[b"<|endoftext|>"]

    for _ in range(max_new_tokens):
        # 输入长度不能超过模型的上下文窗口
        context_ids = ids[-model.context_length:] # 取最近的context length 作为本轮模型的输入
        # tensor shape: [1, ids_length] add []: 增加batch维度
        x = torch.tensor([context_ids], 
            dtype = torch.long, 
            device = device,
            )

        # 只取最后一个位置对下一个token的预测
        logits = model(x)
        next_logits = logits[0, -1, :] # 取出batch中的第一个样本，一次只处理一个prompt

        # temperature: 调整概率分布
        # temperature < 1: 概率更集中，更倾向于选择高分token, 输出通常更稳定
        # temperature = 1: 保持原来的softmax分布
        # temperature > 1: 概率更平缓，低分token更有机会，输出通常更多样
        probabilities = softmax(next_logits / temperature, dim=-1) 

        # Top-p: 保留累计概率达到阈值的最小候选集合
        if top_p < 1.0:
            sorted_probs, sorted_id = torch.sort(probabilities, descending=True)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # 保留使累计概率首次达到/超过 top_p 的那个token
            remove = (cumulative_probs - sorted_probs) >= top_p # 得到>=p的布尔值 [False,False,False,True,True]
            sorted_probs[remove] = 0

            # 重新归一化
            sorted_probs = sorted_probs / sorted_probs.sum()

            #随机抽取索引multinomial返回索引. num_samples表示只抽取一个位置 .item()把索引转化为普通的python： tensor([1]) -> 1
            sampled_position = torch.multinomial(sorted_probs, num_samples = 1).item() 

            next_id = sorted_id[sampled_position].item()
        else:
            next_id = torch.multinomial(probabilities, num_samples=1).item() #p=1,取所有位置的概率

        ids.append(next_id)

        if next_id == eos_id:
            break

    return tokenizer.decode(ids)

def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_files(
        ROOT / "artifacts/tokenizer_ts/vocab.json",
        ROOT / "artifacts/tokenizer_ts/merges.json",
        special_tokens = ["<|endoftext|>"]
    )

    vocab_size = 10000
    context_length = 256

    model = TransformerLM(
        vocab_size = vocab_size,
        context_length = context_length,
        d_model = 128,
        num_layers = 2,
        num_heads = 4,
        d_ff = 344,
        theta = 10000.0,
        device = device,
        dtype = torch.float32,
    )

    checkpoint_path = (ROOT / "checkpoints/tinystories_cosine/step_10000.pt")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9
        )

    print(result)

if __name__ == "__main__":
    main()
