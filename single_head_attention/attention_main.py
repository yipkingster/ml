from jax import random
import jax.numpy as jnp
import jax
import attention

def main():
    print("🧪 Running Scaled Dot Product Attention Tests...\n")
    
    # Setup standard shapes
    BATCH = 2
    SEQ_LEN = 4
    DIM = 8
    
    # Create random inputs (Keys and Values same size for simplicity)
    rng = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(rng, 3)
    
    q = jax.random.normal(k1, (BATCH, SEQ_LEN, DIM))
    k = jax.random.normal(k2, (BATCH, SEQ_LEN, DIM))
    v = jax.random.normal(k3, (BATCH, SEQ_LEN, DIM))
    
    # TEST 1: Basic Shapes
    print("--- Test 1: Shape Check ---")
    out, weights = attention.scaled_dot_product_attention(q, k, v)
    
    print(f"Input Q Shape: {q.shape}")
    print(f"Output Shape:  {out.shape}")
    print(f"Weights Shape: {weights.shape}")
    
    assert out.shape == (BATCH, SEQ_LEN, DIM), "❌ Output shape mismatch!"
    assert weights.shape == (BATCH, SEQ_LEN, SEQ_LEN), "❌ Attention weights shape mismatch!"
    print("✅ Shapes are correct.\n")

    # TEST 2: Masking Logic (The Critical Engineering Check)
    print("--- Test 2: Masking Check ---")
    # Let's say the last word in the sequence is padding.
    # Mask shape: (Batch, 1, 1, Seq) - Standard broadcasting shape
    # 1 = Real Word, 0 = Padding
    mask = jnp.array([[[1, 1, 1, 0]]]) 
    
    _, weights = attention.scaled_dot_product_attention(q, k, v, mask=mask)
    
    # Get the attention weights for the first word in the first batch
    first_word_attn = weights[0, 0, :]
    print(f"Attention weights for word 1: {first_word_attn}")
    
    # The last value (index 3) should be EXACTLY 0.0 because we masked it.
    last_val = first_word_attn[-1]
    print(f"Probability assigned to masked token: {last_val:.6f}")
    
    assert last_val == 0.0, f"❌ Mask failed! Expected 0.0, got {last_val}. Did you use -1e9?"
    print("✅ Masking works (Padding gets 0.0% attention).")

if __name__ == "__main__":
    main()
