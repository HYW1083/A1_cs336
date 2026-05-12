import torch
import numpy.typing as npt
import numpy as np

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    starting_indices = np.random.randint(0, len(dataset)-context_length, batch_size)
    
    input_seqs = []
    target_seqs = []

    for start in starting_indices:
        input_seq = dataset[start: start + context_length]    # need to notice the slice is [ , ). cannot get the latter index.
        target_seq = dataset[start + 1: start + context_length +  1]

        input_seqs.append(input_seq)
        target_seqs.append(target_seq)

    # stack them together first, then convert to tensor
    input_batch = torch.tensor(np.stack(input_seqs),dtype = torch.long, device = device)
    target_batch = torch.tensor(np.stack(target_seqs), dtype = torch.long, device = device)

    return input_batch, target_batch

    



