import torch 
import math
from einops import einsum, rearrange
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
            token_positions: Integer positions for each token, shape `(..., seq_len)` or `(seq_len,)`

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


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
        use_rope: bool = False,
    ):
        """
        Causal multi-head self-attention.

        Args:
            d_model: Input and output embedding dimension.
            num_heads: Number of attention heads. `d_model` must divide evenly by this.
            max_seq_len: Maximum sequence length for RoPE cache, required if `use_rope=True`.
            theta: RoPE base frequency, required if `use_rope=True`.
            device: Device for parameters and RoPE cache.
            dtype: Parameter dtype.
            use_rope: Whether to apply RoPE to queries and keys.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if use_rope and (max_seq_len is None or theta is None):
            raise ValueError("max_seq_len and theta are required when use_rope=True")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if use_rope: 
            self.rope = RoPE(theta, self.d_head, max_seq_len, device=device)
        else:
            self.rope = None
    
    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        """
        Args:
            x: Input tensor of shape `(..., sequence_length, d_model)`.
            token_positions: Optional token positions for RoPE. Usually shape
                `(sequence_length,)` or broadcastable to `x.shape[:-1]`.

        Returns:
            Tensor of shape `(..., sequence_length, d_model)`.
        """
        seq_len = x.shape[-2]

        # q = self.q_proj(x)
        # k = self.k_proj(x)
        # v = self.v_proj(x)
        # or stretch goal:

        qkv_proj = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        qkv = x @ qkv_proj.T
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "... seq (h d) -> ... h seq d", h=self.num_heads) # (batch, heads, seq_len, d_head)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h=self.num_heads)

        if self.rope is not None:
            if token_positions is not None and token_positions.ndim == q.ndim - 2:
                token_positions = token_positions.unsqueeze(-2)   # shape: (batch_size, seq_len) -> (batch_size, 1, seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        causal_mask = ~causal_mask

        output = scaled_dot_product_attention(q, k, v, causal_mask)
        output = rearrange(output, "... h seq d -> ... seq (h d)")
        return self.output_proj(output)

class TransformerBlock(torch.nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None):
        """
        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention. 
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache.
            theta (float): RoPE parameter.
            device: Device on which to store the module parameters and buffers.
            dtype: Data type of the module parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.ln1 = RMSNorm(d_model,device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype, use_rope=True)
        self.ln2 = RMSNorm(d_model,device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape `(..., seq_len, d_model)`.
            token_positions: Optional token position for RoPE.

        Returns:
            Tensor of shape `(..., seq_len, d_model)`.
        """
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))

        return x


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device=None,
        dtype=None):
        """
        Args:
            vocab_size (int): The number of unique items in the output vocabulary to be predicted.
            context_length (int): The maximum number of tokens to process at once.
            d_model (int): The dimensionality of the model embeddings and sublayer outputs.
            num_layers (int): The number of Transformer layers to use.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
            rope_theta (float): The RoPE $\\Theta$ parameter.
            device: Device on which to store the module parameters and buffers.
            dtype: Data type of the module parameters.
        """
        super().__init__()
        self.context_length = context_length
        
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)

        self.layers = torch.nn.ModuleList(
            [TransformerBlock(d_model=d_model, 
                             num_heads=num_heads, 
                             d_ff=d_ff, 
                             max_seq_len=context_length, 
                             theta=theta, 
                             device=device, 
                             dtype=dtype) 
            for _ in range(num_layers)])
        
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        return the output of running a forward pass on the input indices.
        """
        seq_len = in_indices.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("Input sequence length exceeds context_length")
        
        x = self.token_embeddings(in_indices)
        token_positions = torch.arange(seq_len, device=in_indices.device)

        for layers in self.layers:
            x = layers(x, token_positions=token_positions)

        x = self.ln_final(x)
        return self.lm_head(x)
    


