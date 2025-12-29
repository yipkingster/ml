from flax import linen as nn
import encoder_layer

class Encoder(nn.Module):
    vocab_size: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int  # How many layers to stack (e.g., 6)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask, training: bool):
        # Input x is currently token IDs: (Batch, Seq) e.g., [[101, 25, ...]]
        
        # 1. Embedding Layer
        # Convert IDs to Vectors. Use nn.Embed(num_embeddings, features)
        # Initialized randomly.
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)
        
        # (Optional: Add Positional Encoding here - we will skip implementation for now 
        # and just assume 'x' has position info, or you can just add zeros)
        
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