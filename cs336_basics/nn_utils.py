import numpy as np
import torch
from torch import Tensor
import math
from collections.abc import Iterable
from jaxtyping import Bool, Float, Int

def softmax(x: torch.Tensor, dim: int) -> Float[Tensor, "..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim` of input.

    Args:
        x (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `x` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of same shape as `x` with the output of softmax normalizing the specified `dim`.
    """
    max_val = torch.max(x, dim, keepdim=True).values # output of torch.max is values and indices, need to specify values. keep same dim with input, so it should be True, otherwise delete the last dim
    x_stable = x - max_val                   # because keepdim, do not need to consider dims
    x_exp = torch.exp(x_stable)
    output = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

    return output

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class. 
        Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # extract logits corresponding to the targets class 
    targets_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)) # make sure index has same dim with inputs, unsqueeze and squeeze only operate the dim of shape = 1

    # do not use softmax as middle process for the numerical stability
    # not torch.log(torch.sum(torch.exp(inputs), dim=-1)) because exp(inputs) would be overflowed, log-sum-exp trick for numerical stability by subtracting the largest element
    logsumexp = torch.logsumexp(inputs, dim=-1, keepdim=True) 

    loss = torch.mean(-targets_logits + logsumexp)
    return loss

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    
    params_with_grad = []            # params_with_grad = [parameter for parameter in parameters if parameter.grad is not None]
    for parameter in parameters:
        if parameter.grad is not None:
            params_with_grad.append(parameter)
    
    if not params_with_grad:     # if this list is Null, the bool is False, need to notice if not params_with_grad: not equal to if params_with_grad is None: 
        return                 # because the former is the bool([]) the latter could be [] (only check the variable is None or not), here need to check the False value, including None,[],0,{},False,etc.

    total_squared_norm = 0                                                    # total_norm = torch.sqrt(
    for parameter in params_with_grad:                                        #     sum(torch.sum(parameter.grad.detach() ** 2) for parameter in params_with_grad)
        total_squared_norm += torch.sum(parameter.grad ** 2)                  # )
    
    total_norm = torch.sqrt(total_squared_norm)

    clip_coef = max_l2_norm / (total_norm + 1e-6)

    if max_l2_norm < total_norm:
        for parameter in params_with_grad:
            parameter.grad.mul_(clip_coef)            # modify in place(using mul_()) "_" represents in-place change; do not use parameter.grad = clip_coef * parameter.grad


