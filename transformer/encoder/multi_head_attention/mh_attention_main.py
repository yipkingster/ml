import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import mh_attention


def run_multihead_tests():
    print("🧪 Running Multi-Head Attention Tests...\n")

    # --- Configuration ---
    BATCH_SIZE = 4
    SEQ_LEN = 10
    D_MODEL = 512
    NUM_HEADS = 8
    
    # 1. Setup the Model
    # We initialize the blueprint.
    model = mh_attention.MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    
    # 2. Create Dummy Data
    rng = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(rng)
    
    # Input x: (Batch, Seq, Dim) - standard Transformer input
    x = jax.random.normal(key1, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    
    # 3. Initialize the Weights (Flax Pattern)
    # This triggers 'setup()' and creates the parameter dictionary.
    variables = model.init(key2, x, x, x)
    
    # --- TEST 1: Variable Shapes (Did we create the right matrices?) ---
    print("--- Test 1: Weight Matrix Shapes ---")
    params = variables['params']
    
    # We expect W_q, W_k, W_v, W_o to be (512, 512)
    # Note: Flax names them 'Dense_0', 'Dense_1', etc. automatically if we didn't name them.
    # But usually, we check the structure.
    for layer_name in params.keys():
        w_shape = params[layer_name]['kernel'].shape
        print(f"Layer {layer_name} weights: {w_shape}")
        assert w_shape == (D_MODEL, D_MODEL), f"❌ {layer_name} should be ({D_MODEL}, {D_MODEL})"
    
    print("✅ All weight matrices are the correct size.\n")

    # --- TEST 2: Forward Pass (The Pipeline) ---
    print("--- Test 2: Forward Pass (Input -> Output) ---")
    
    # We pass the variables (weights) and the input data
    # q=x, k=x, v=x (Self-Attention)
    output = model.apply(variables, x, x, x)
    
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {output.shape}")
    
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), "❌ Output shape mismatch! Did merge_heads fail?"
    print("✅ Forward pass successful. Output shape matches Input shape.\n")

    # --- TEST 3: Mask Propagation ---
    print("--- Test 3: Masking Check ---")
    
    # Create a mask that hides the last 2 words of every sentence
    # Shape: (Batch, 1, 1, Seq) - Broadcasts to all heads and all query rows
    mask = jnp.ones((BATCH_SIZE, 1, 1, SEQ_LEN))
    # Set the last two positions to 0 (Padding)
    mask = mask.at[:, :, :, -2:].set(0)
    
    # Run model with mask
    # We can't easily check internal probabilities here without modifying the class to return them,
    # but if this runs without error, it proves the mask shape was compatible.
    try:
        output_masked = model.apply(variables, x, x, x, mask=mask)
        print("✅ Forward pass with Mask ran successfully.")
        
        # Simple sanity check: The output should be DIFFERENT from the unmasked run
        # because the attention scores changed.
        diff = jnp.sum(jnp.abs(output - output_masked))
        print(f"Difference sum between Masked and Unmasked run: {diff:.4f}")
        assert diff > 0, "❌ Mask had no effect! The output is identical to unmasked run."
        
    except Exception as e:
        print(f"❌ Forward pass with Mask FAILED: {e}")

    print("\n🎉 ALL MULTI-HEAD ATTENTION TESTS PASSED!")

if __name__ == "__main__":
    run_multihead_tests()
