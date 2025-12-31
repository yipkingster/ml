from flax import linen as nn
from jax import numpy as jnp
import encoder_layer
from positional_encoding import pos_encoding as pe
class Encoder(nn.Module):
    vocab_size: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int  # How many layers to stack (e.g., 6)
    dropout_rate: float = 0.1
    max_len: int = 5000

    @nn.compact
    def __call__(self, x, mask, training: bool):
        # Input x is currently token IDs: (Batch, Seq) e.g., [[101, 25, ...]]
        
        # 1. Embedding Layer
        # Convert IDs to Vectors. Use nn.Embed(num_embeddings, features)
        # Initialized randomly.
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)
        # "In the embedding layers, we multiply those weights by sqrt(d_model)"
        # - Vaswani et al.
        x = x * jnp.sqrt(self.d_model)
        # Add Positional Encoding.
        x = pe.PositionalEncoding(self.d_model, self.max_len)(x)

        # 2. Dropout on Embeddings
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        
        # 3. The Stack of Layers
        # Loop 'self.num_layers' times.
        for i in range(self.num_layers):
            # Create an EncoderLayer. 
            # Note: In Flax, calling the class inside the loop with a unique name 
            # (or letting Flax handle naming) creates a SEPARATE set of weights for each layer.
            layer = encoder_layer.EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
            x = layer(x, mask, training)
        
        # 4. Standard practice to do normalization.
        x = nn.LayerNorm()(x)
        return x