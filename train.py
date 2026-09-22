from pathlib import Path

import numpy as np
import torch

from tqdm.auto import tqdm

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import save_checkpoint, load_checkpoint

ROOT = Path(__file__).resolve().parent

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train_data = np.memmap(
    #     ROOT / "data/tokenized/tinystories_train.bin",
    #     dtype = "<u2",
    #     mode = "r",
    # )
    # data/tokenized

    # valid_data = np.memmap(
    #     ROOT / "data/tokenized/tinystories_valid.bin",
    #     dtype = "<u2",
    #     mode = "r",
    # )

    # checkpoint_dir = ROOT / "checkpoints/tinystories_cosine"
    # checkpoint_dir.mkdir(parents=True, exist_ok=True) # parents=True：如果上级目录 checkpoints 不存在，也一起创建。 exist_ok=True：如果目标文件夹已经存在，不报错。

    train_data = np.memmap(
        ROOT / "data/tokenized/owt_train.bin",
        dtype = "<u2",
        mode = "r",
    )

    valid_data = np.memmap(
        ROOT / "data/tokenized/owt_valid.bin",
        dtype = "<u2",
        mode = "r",
    )

    checkpoint_dir = ROOT / "checkpoints/owt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    vocab_size = 32000
    batch_size = 20
    context_length = 512

    model = TransformerLM(
        vocab_size = vocab_size,
        context_length = context_length,
        d_model = 512,
        num_layers = 8,
        num_heads = 8,
        d_ff = 1408,
        theta = 10000.0,
        device = device,
        dtype = torch.float32,
    )

    num_steps = 200000
    max_learning_rate = 3e-4
    min_learning_rate = 3e-5
    warmup_iters = max(1, int(num_steps * 0.02))
    cosine_cycle_iters = num_steps

    optimizer = AdamW(
        model.parameters(),
        lr = max_learning_rate, #先给lr赋一个初值，实际更新时替换
        weight_decay = 0.1,
    )

    resume_path = None
    # resume_path = ROOT / "checkpoints/tinystories/step_400.pt"
    start_step = 0

    if resume_path is not None:
        start_step = load_checkpoint(
            src=resume_path,
            model=model,
            optimizer=optimizer,
        )

        print(f"Resumed from checkpoint: {resume_path}")
        print(f"Completed training steps: {start_step}")

    # initial_valid_loss = evaluate(
    #     model=model,
    #     dataset=valid_data,
    #     batch_size=batch_size,
    #     context_length=context_length,
    #     device=device,
    # )
 
    # print(f"Initial validation loss: {initial_valid_loss:.4f}")

    model.train()

    progress_bar = tqdm(
        range(start_step, num_steps),
        total=num_steps,
        initial=start_step,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        )

    for step in progress_bar:
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
            )
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

            # optimizer.param_groups = [
            #     {
            #         "params": [模型参数1, 模型参数2, ...],
            #         "lr": 3e-4,
            #         "weight_decay": 0.1,
            #         # 还有 betas、eps 等配置
            #     }
            # ]

        # 每步重新随机抽取一个batch:
        x, y = get_batch(
        dataset = train_data,
        batch_size = batch_size,
        context_length = context_length,
        device = device,
        )

        optimizer.zero_grad(set_to_none = True) # zero_grad: 清除旧梯度。 set_to_none = True，表示当前没有梯度（默认）。set_to_none = False：已有梯度张量被填成零
        # 混合精度
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)

        loss = cross_entropy(
            logits.float().reshape(-1, vocab_size), # -1 表示让 PyTorch 自动计算该维度的大小: (batch_size, context_length, vocab_size) -> (batch_size*context_length, vocab_size)
            y.reshape(-1), # target: (batch_size, context_length) -> (batch_size*context_length，)
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError("Training loss is NaN or infinite.")

        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm = 1.0)
        optimizer.step()

        if step == start_step or (step + 1) % 50 == 0:
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{lr:.2e}",
            )
        
        if (step + 1) % 10000 == 0 or step + 1 == num_steps:
            checkpoint_path = checkpoint_dir / f"step_{step + 1}.pt"

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step + 1,
                out = checkpoint_path,
            )

            tqdm.write(f"Checkpoint saved: {checkpoint_path}")

        if (step + 1) % 1000 == 0:
            valid_loss = evaluate(
                model=model,
                dataset=valid_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )

            tqdm.write(
                f"Step{step + 1}/{num_steps} | "
                f"Validation loss: {valid_loss:.4f}"
                )

        # print(f"Input shape: {tuple(x.shape)}")
        # print(f"Target shape: {tuple(y.shape)}")
        # print(f"Logits shape: {tuple(logits.shape)}")
        # print(f"Loss shape: {tuple(loss.shape)}")
        # print(f"Loss: {loss.item():.4f}")
    print(f"Training completed.")


@torch.no_grad()
def evaluate(
    model,
    dataset,
    batch_size,
    context_length,
    device,
    num_batches = 100,
):
    # 记录进入评估前的模式
    was_training = model.training
    model.eval()

    # 保存随机状态，避免评估改变后序训练的采样顺序
    rng_state = np.random.get_state()
    np.random.seed(123)
    total_loss = 0
    try:
        for _ in range(num_batches):
            x, y = get_batch(
                dataset = dataset,
                batch_size = batch_size,
                context_length = context_length,
                device = device,
            )
            
            logits = model(x)

            loss = cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
            )

            total_loss += loss.item()

    finally:
        np.random.set_state(rng_state)
        model.train(was_training)

    return total_loss / num_batches
if __name__ == "__main__":
    main()