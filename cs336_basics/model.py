import torch 
import math
from einops import einsum
from torch import Tensor
from cs336_basics.nn_utils import softmax
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

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Args:
            theta (float): Θ value for the RoPE.
            d_k (int): Embedding dimension size for the query or key tensor.
            max_seq_len (int): Maximum sequence length that will be input.
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE") # make sure they can do 2d coordination rotations so it must be even.
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        rotation_matrix = self.generate_rotation_matrix(theta, max_seq_len, d_k, device=device)
        self.register_buffer("rotation_matrix", rotation_matrix, persistent=False) # avoid re-calculation everytime when calls forward(), directly lookup this table
        

    def generate_rotation_block(self, theta: float, block_index: int, seq_pos: int, d_k: int, device=None) -> torch.Tensor:
        """
        Generate the 2x2 RoPE rotation block for one pair of hidden dimensions.

        Args: 
            theta: RoPE base frequency parameter.
            block_index: Zero-based index of the 2D block along the hidden dimensions.
            seq_pos: Token position i in the sequence.
            d_k (int): Embedding dimension size for the query or key tensor.
            device: Device to create the returned tensor.
        
        Returns:
            Return the 2x2 rotation matrix used by RoPE for a single pair of dimensions.
        """
        angle = torch.tensor(seq_pos / (theta ** (2 * block_index / d_k)))
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        rotation_block = torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])]) # because cos and sin are already tensor, do not need to reconstruct a new Tensor.
        return rotation_block

    def generate_rotation_matrix(self, theta: float, max_seq_len: int, d_k: int, device=None) -> torch.Tensor:
        """
        Precompute the RoPE rotation matrix for every possible token position.

        For each position `pos` in `[0, max_seq_len)`, this builds a block-diagonal matrix of shape `(d_k, d_k)`. Each 2x2 block rotates one
        pair of hidden dimensions according to the RoPE angle for that position and dimension pair. 
        
        Args:
            theta: RoPE base frequency parameter.
            max_seq_len: Maximum sequence length to precompute rotations for.
            d_k: Query/Key hidden dimension. Must be even. 两两一组做二维旋转例如：
                (x_0, x_1)
                (x_2, x_3)
                (x_4, x_5)
                ...
                每一组用一个 2x2 旋转矩阵：
                [sin   cos]
                所以 d_k 必须能被 2 整除。
            device: Device on which to create the rotation table.

        Returns:
            A tensor of shape `(max_seq_len, d_k, d_k)`, where `rotation_matrix_table[pos]` is the full RoPE rotation matrix for token position `pos`.
        """
        rotation_matrix_table = torch.zeros(max_seq_len, d_k, d_k, device=device, dtype=torch.float32)
        # RoPE rotates each token's query/key vector according to its position, then computation of attention score can get real results of relative position.
        for pos in range(max_seq_len):
            blocks = [self.generate_rotation_block(theta, k, seq_pos=pos, d_k=d_k, device=device) for k in range(d_k // 2)]
            rotation_matrix_table[pos] = torch.block_diag(*blocks)

        return rotation_matrix_table

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE rotations to query or key vectors.
        
        Args:
            x: Input query/key tensor of shape `(..., seq_len, d_k)`.
        
        Returns:
            A tensor with the same shape as `x`, where each vector `x[..., i, :]` has been rotated using RoPE matrix corresponding to its token position.
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
        
        token_positions = token_positions.to(x.device)
        rotation_matrix = self.rotation_matrix[token_positions] 
        x_rotated = torch.matmul(rotation_matrix, x.unsqueeze(-1)).squeeze(-1)
        return x_rotated.to(in_type)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf")) # fill the mask false value with -inf

    attention_weights = softmax(scores, dim=-1)
    output = einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return output
