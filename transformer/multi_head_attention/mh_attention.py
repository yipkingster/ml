from flax import linen as nn
import jax.numpy as jnp
import single_head_attention.attention as sh_attention

class MultiHeadAttention(nn.Module):
    d_model: int   # Total dimension (e.g., 512)
    num_heads: int # Number of heads (e.g., 8)

    def setup(self):
        # 1. Validation
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model // self.num_heads

        # 2. Define the Learnable Layers (The Weights)
        # Note: We use one big matrix for all heads combined, then split them later.
        self.W_q = nn.Dense(self.d_model)
        self.W_k = nn.Dense(self.d_model)
        self.W_v = nn.Dense(self.d_model)
        self.W_o = nn.Dense(self.d_model)

    def split_heads(self, x):
        """
        Input x: (Batch, Seq, d_model)
        Ouput:    (Batch, Heads, Seq, d_k)
        """
        batch_size = x.shape[0]
        # Split the d_model to d_k and get one more head dimension.
        x =x.reshape(batch_size, -1, self.num_heads, self.d_k)
        # Alternative: specify seq_size and use jnp.reshape()
        # seq_size = x.shape[1]
        # x = jnp.reshape(batch_size, seq_size, self.num_heads, self.d_k)

        # Transpose the head dimension out of the last 2. The last 2 should be seq and d_k so matmul can apply to them.
        # Tranpose [batch, seq, head, d_k] into [batch, head, seq, d_k]
        x = x.transpose(0, 2, 1, 3)
        # Alternative:
        # x = jnp.transpose(x, (0, 2, 1, 3))
        return x

    def merge_heads(self, x):
        """
        Input x: (Batch, Heads, Seq, d_k)
        Output:    (Batch, Seq, d_model)
        """
        batch_size = x.shape[0]
        # Reshape only works if the 2 merged dimensions are next to each other. So we need to transpose first.
        # Transpose [batch, head, seq, d_k] into [batch, seq, head, d_k]
        x = x.transpose(0, 2, 1, 3)
        seq_size = x.shape[1]
        # Now reshap/merge the last 2.
        x = x.reshape(batch_size, seq_size, self.d_model)
        return x

    def __call__(self, q_in, k_in, v_in, mask=None):
        # 1. Linear Projections (Learnable Weights)
        q = self.W_q(q_in)
        k = self.W_k(k_in)
        v = self.W_v(v_in)

        # 2. Split Heads (Transform 3D to 4D)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 3. Scaled Dot Product Attention
        # Note: scaled_dot_product_attention handles 4D input naturally
        attn_out, _ = sh_attention.scaled_dot_product_attention(q, k, v, mask)

        # 4. Merge Heads (Transform 4D back to 3D)
        output = self.merge_heads(attn_out)

        # 5. Final Linear Projection
        output = self.W_o(output)
        
        return output