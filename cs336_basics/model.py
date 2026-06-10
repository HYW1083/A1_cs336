import torch 
import math
from einops import einsum
from torch import Tensor
from jaxtyping import Bool, Float, Int

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Args:
            in_features: int final dimension of input 
            out_features: out final dimension of output
            device: torch.device| None = None Device to store the parameters on
            dtype: torch.dtypte | None = None Data type of the parameters

        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3*std,
            b=3*std
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Args:
            num_embeddings: size of the vocabulary
            embedding_dim: Dimension of the embedding vectors, i.e., d_model
            device: torch.device| None = None Device to store the parameters on
            dtype: torch.dtypte | None = None Data type of the parameters
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding_matrix = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(
            self.embedding_matrix,
            mean=0,
            std=1,
            a=-3,
            b=3
        )
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        Args:
            token_ids: (batch_size, sequence_length)
            output: (batch_size, sequence_length, embedding_dim)
        Returns:
            embeddings for given token IDs
        """
        # advanced indexing: lookup for each token id in embedding_matrix, because embedding_matrix has emb_dim, so the output has emb_dim
        return self.embedding_matrix[token_ids]   
        # other way to write: 
        # batch_size, sequence_length = token_ids.shape
        # output = torch.empty(batch_size, sequence_length, self.embedding_dim)

        # for i, seq in enumerate(token_ids):
        #     for j, token_id in enumerate(seq):
        #         output[i][j] = self.embedding_matrix[token_id]
        # return output

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gains = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.gains
        return result.to(in_dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x) 

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> Float[Tensor, " ... d_model"]:
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally to swiglu.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        """ 
        Implement the SwiGLU feedforward network
        Args:
            x(Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer
        Returns:
            Float[Tensor, " ... d_model"]: Output embeddings of the same shape as the input embeddings
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))

        
