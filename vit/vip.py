# Vision Transformer pipeline.
import jax
import jax.numpy as jnp
from flax import linen as nn

class VisionTransformer(nn.Module):
    embed_dim: int       # e.g., 512
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
        # Hint: Use nn.Conv.
        # - The output features should be self.embed_dim.
        # - The kernel and stride must BOTH be self.patch_size to avoid overlap.
        x = ... 
        
        # ---------------------------------------------------------
        # 2. Flatten spatial dimensions
        # ---------------------------------------------------------
        # Current shape: [Batch, H_grid, W_grid, Embed_Dim]
        # Target shape:  [Batch, Num_Patches, Embed_Dim]
        b, h, w, c = x.shape
        x = x.reshape(...)
        
        # ---------------------------------------------------------
        # 3. Add [CLS] Token
        # ---------------------------------------------------------
        # Step A: Create the learnable parameter. 
        # Hint: Shape is (1, 1, embed_dim). Initialize with zeros or normal.
        cls_token = self.param('cls_token', nn.initializers.zeros, (...))
        
        # Step B: Broadcast it to the batch size.
        # Hint: You need one CLS token per image in the batch. 
        # Use jnp.tile to repeat it 'b' times.
        cls_token = ...
        
        # Step C: Attach it to the start of the sequence.
        # Hint: jnp.concatenate along axis 1 (the sequence axis).
        x = ...
        
        # ---------------------------------------------------------
        # 4. Add Position Embeddings
        # ---------------------------------------------------------
        # Step A: Create the learnable parameter.
        # Hint: This must match the total sequence length (Num_Patches + 1 for CLS).
        # Shape: (1, Total_Seq_Len, Embed_Dim)
        num_patches = (self.patch_size ** 2) # Wait, is this right? Check logic below.
        total_seq_len = x.shape[1]
        pos_embedding = self.param('pos_embed', nn.initializers.normal(0.02), (...))
        
        # Step B: Add it to x. (Broadcasting handles the batch dim automatically)
        x = ...
        
        # ---------------------------------------------------------
        # 5. Transformer Encoder (The part you already built!)
        # ---------------------------------------------------------
        # We can just loop through your existing EncoderBlock
        # (Assuming you have an EncoderBlock class defined elsewhere)
        for _ in range(self.num_layers):
            # x = EncoderBlock(...)(x, train=train) # Uncomment when ready
            pass # Placeholder
            
        # ---------------------------------------------------------
        # 6. Classification Head
        # ---------------------------------------------------------
        # Step A: Extract the [CLS] token result.
        # Hint: It was at index 0. We want the vector for every item in batch.
        # Shape change: [Batch, Seq_Len, Dim] -> [Batch, Dim]
        cls_final = ...
        
        # Step B: Project to logits (Num Classes)
        # Hint: Use nn.Dense
        logits = ...
        
        return logits