# Vision Transformer pipeline.
import jax
import jax.numpy as jnp
from flax import linen as nn
from ..transformer.encoder import encoder_layer

D_MODEL = 512
D_FF = 2048
DROPOUT_RATE = 0.1

# This module will "translate" visual pixel representations into a sequence of
# tokens with semantic meaning.
class VisionTransformer(nn.Module):
    embed_dim: int       # Dimension of the ouput features. e.g., 512
    hidden_dim: int      # e.g., 2048 (for MLP)
    num_heads: int       # e.g., 8
    num_layers: int      # e.g., 6
    num_classes: int     # e.g., 10 (CIFAR-10)
    patch_size: int      # e.g., 16 (16x16 pixels)
    
    @nn.compact
    def __call__(self, x, train=True):
        # Input x shape: [Batch, Height, Width, Channels] (e.g., B, 224, 224, 3)
        
        # ---------------------------------------------------------
        # 1. Patch Embedding (The "Conv2d" Trick)
        # ---------------------------------------------------------
        # - The output features should be self.embed_dim.
        # - The kernel and stride must BOTH be self.patch_size to avoid overlap.
        # - padding = "VALID" will just ignore extra rows/columns that can not
        # fit into the multiples of "strides".
        x = nn.Conv(self.embed_dim, (self.patch_size, self.patch_size),
                    (self.patch_size, self.patch_size), padding="VALID")
        
        # ---------------------------------------------------------
        # 2. Flatten spatial dimensions
        # ---------------------------------------------------------
        # Current shape: [Batch, H_grid, W_grid, Embed_Dim]
        # Target shape:  [Batch, Num_Patches, Embed_Dim]
        b, h, w, c = x.shape
        x = x.reshape(b, -1, c)
        
        # ---------------------------------------------------------
        # 3. Add [CLS] Token
        # ---------------------------------------------------------
        # Step A: Create the learnable parameter. 
        # Shape is (1, 1, embed_dim). Initialize with zeros or normal.
        cls_token = self.param('cls_token', nn.initializers.zeros,
                               (1, 1, self.embed_dim))
        
        # Step B: Broadcast it to the batch size.
        # You need one CLS token per image in the batch. 
        # Use jnp.tile to repeat it 'b' times.
        cls_token = jnp.tile(cls_token, (b, 1, 1))
        
        # Step C: Attach it to the start of the sequence.
        # Hint: jnp.concatenate along axis 1 (the sequence axis).
        x = jnp.concatenate((cls_token, x), axis=1)
        
        # ---------------------------------------------------------
        # 4. Add Position Embeddings
        # ---------------------------------------------------------
        # Step A: Create the learnable parameter.
        # This must match the total sequence length (Num_Patches + 1 for CLS).
        # Shape: (1, Total_Seq_Len, Embed_Dim)
        num_patches = (self.patch_size ** 2) # Wait, is this right? Check logic below.
        total_seq_len = x.shape[1]
        pos_embedding = self.param('pos_embed', nn.initializers.normal(0.02),
                                   (1, total_seq_len, self.embed_dim))
        
        # Step B: Add it to x. (Broadcasting handles the batch dim automatically)
        x = x + pos_embedding
        
        # ---------------------------------------------------------
        # 5. Transformer Encoder (The part you already built!)
        # ---------------------------------------------------------
        # We can just loop through your existing EncoderBlock
        # (Assuming you have an EncoderBlock class defined elsewhere)
        for _ in range(self.num_layers):
            layer = encoder_layer.EncoderLayer(D_MODEL, self.num_heads, D_FF, DROPOUT_RATE)
            x = layer(x, None, train)
            
        # ---------------------------------------------------------
        # 6. Classification Head
        # ---------------------------------------------------------
        # Step A: Extract the [CLS] token result.
        # It was at index 0. We want the vector for every item in batch.
        # Shape change: [Batch, Seq_Len, Dim] -> [Batch, Dim]
        # Or: cls_final = cla_final.squeenze(axis=1)
        # jnp automatically removes dimension with size 0
        cls_final = x[:, 0, :] 

        
        # Step B: Project to logits (Num Classes)
        # Solve the weights of embed_dim=512 and bias of num_classes.
        logits = nn.Dense(self.num_classes)(cls_final)
        
        return logits