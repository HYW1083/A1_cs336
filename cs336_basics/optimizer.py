import math
import torch
from collections.abc import Callable
from typing import Optional

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters: 
        return (it / warmup_iters) * max_learning_rate

    elif it > cosine_cycle_iters:
        return min_learning_rate

    else:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(progress*math.pi)) * (max_learning_rate - min_learning_rate)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)     # call initial functions in Optimizer, it can process params as group, etc.

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()  # closure can recalcuate loss 
        for group in self.param_groups:
            lr = group["lr"]   # get lr and params from the dictionary of group
            for p in group["params"]:
                if p.grad is None:
                    continue    # skip some params aren't involved this calculation
            
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
                
        return loss

if __name__ == "__main__":   # used for debug, uv run python optimizer.py only executes this
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=0.001)

    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item()) 
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.


class AdamW(torch.optim.Optimizer):
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    def __init__(
        self, 
        params,
        lr = 1e-3, 
        betas = (0.9, 0.999), 
        eps = 1e-8, 
        weight_decay = 0.1):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
            
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]    
                m = state["m"]
                v = state["v"]

                grad = p.grad.data
                lr_adjusted = lr * (math.sqrt(1 - beta2 ** t))/(1 - beta1 ** t)
                p.data -= lr * weight_decay * p.data
                m.mul_(beta1).add_(grad, alpha = 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                p.data -= lr_adjusted * m / (torch.sqrt(v) + eps)
        return loss




                

                


            







        


    




