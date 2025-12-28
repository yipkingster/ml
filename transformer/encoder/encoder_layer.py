from flax import linen as nn
from multi_head_attention import mh_attention as mh_attention
from feed_forward import feed_forward

class EncoderLayer(nn.Module):
    '''Encoder layer in the Transformer architecture.
    
    Encoder layer encapsulates attention and FFN with residual connection and regularization (dropout) in between.
    This layer will repeat uniformly for all attention output.

    Attributes:
        d_model:        Dimension of the model, usually 512
        num_heads:      Number of attention heads to capture different aspects of the semantic meaning. Usually 8
        d_ff:           Dimension of the inner FFN layer, usually 4 x d_model = 4x512 = 2048
        dropout_rate:   How many node in the layer will be turned out randomly. Regularization technique to prevent overfitting.
    '''
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask, training: bool):
        # --- PART 1: ATTENTION ---
        # 1. Create MultiHeadAttention module
        attn = mh_attention.MultiHeadAttention(self.d_model, self.num_heads)
        
        # 2. Run Attention (Remember to pass the mask!)
        attn_out = attn(x, x, x, mask)
        
        # 3. Apply Dropout to attn_out
        attn_out = nn.Dropout(self.dropout_rate, deterministic=not training)(attn_out)
        
        # 4. Residual Connection + Norm
        # Add the original 'x' to 'attn_out'
        # Then apply nn.LayerNorm()
        x_norm1 = x + attn_out
        # Since this layer is included in @nn.compact decorator, master init() of this class
        # will call init() automatically for it and all other children modules.
        x_norm1 = nn.LayerNorm()(x_norm1)

        # --- PART 2: FEED-FORWARD ---
        # 5. Create PositionwiseFeedForward module
        ffn = feed_forward.PositionwiseFeedForward(self.d_model, self.d_ff)
        
        # 6. Run FFN
        ffn_out = ffn(x_norm1)
        
        # 7. Apply Dropout to ffn_out
        ffn_out = nn.Dropout(self.dropout_rate, deterministic=not training)(ffn_out)
        
        # 8. Residual Connection + Norm
        # Add 'x_norm1' (the input to this section) to 'ffn_out'
        # Then apply nn.LayerNorm()
        x_final = x_norm1 + ffn_out
        x_final = nn.LayerNorm()(x_final)
        
        return x_final