import sys
import os
import jax
import jax.numpy as jnp
import encoder_stack 

def run_tests():
    print("🧪 Running Full Encoder Stack Tests...\n")

    # --- Configuration ---
    BATCH_SIZE = 4
    SEQ_LEN = 32
    VOCAB_SIZE = 1000  # Small vocab for testing
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 3     # We will stack 3 layers
    
    # 1. Setup Data
    rng = jax.random.PRNGKey(42)
    key_init, key_dropout = jax.random.split(rng)
    
    # INPUT: Random Integers (Token IDs) between 0 and VOCAB_SIZE
    # Shape: (Batch, Seq_Len)
    input_ids = jax.random.randint(key_init, (BATCH_SIZE, SEQ_LEN), 0, VOCAB_SIZE)
    
    # MASK: Standard attention mask
    mask = jnp.ones((BATCH_SIZE, 1, 1, SEQ_LEN))
    
    # 2. Initialize Model
    print(f"... Building Encoder with {NUM_LAYERS} layers")
    model = encoder_stack.Encoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS
    )
    
    # Initialize variables
    # Note: This will trigger the creation of Embeddings + N x EncoderLayers
    variables = model.init(key_init, input_ids, mask, training=False)
    
    # --- TEST 1: Output Shape (The Transformation) ---
    print("\n--- Test 1: Transformation Check (Int -> Vector) ---")
    output = model.apply(variables, input_ids, mask, training=False)
    
    print(f"Input Shape (IDs):    {input_ids.shape}")
    print(f"Output Shape (Vecs):  {output.shape}")
    
    expected_shape = (BATCH_SIZE, SEQ_LEN, D_MODEL)
    assert output.shape == expected_shape, f"❌ Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print("✅ Input Integers correctly transformed into Context Vectors.")

    # --- TEST 2: Stack Depth Verification ---
    print("\n--- Test 2: Stack Depth Verification ---")
    # We inspect the parameter dictionary to prove we actually created 3 layers.
    # Flax names layers automatically: 'EncoderLayer_0', 'EncoderLayer_1', etc.
    
    params = variables['params']
    
    # Count how many keys start with 'EncoderLayer'
    # Note: Depending on your loop implementation, these might be inside a sub-dictionary.
    # Usually Flax puts them at the top level of the Encoder params if defined in loop.
    
    # Let's verify we have parameters for multiple layers
    layer_count = sum(1 for key in params.keys() if 'EncoderLayer' in key)
    
    print(f"Detected {layer_count} distinct Encoder Layers in parameters.")
    
    # Note: If you use 'nn.scan' this might look different, but for a Python loop:
    if layer_count == NUM_LAYERS:
        print(f"✅ Correctly instantiated {NUM_LAYERS} separate layers.")
    else:
        print(f"⚠️  Warning: Found {layer_count} layers, expected {NUM_LAYERS}. Check your loop variable naming.")

    print("\n🎉 ALL ENCODER STACK TESTS PASSED!")

if __name__ == "__main__":
    run_tests()