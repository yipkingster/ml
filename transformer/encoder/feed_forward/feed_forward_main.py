import jax
import feed_forward

def run_ffn_tests():
    print("🧪 Running Position-wise FFN Tests...\n")

    # --- Configuration ---
    BATCH_SIZE = 4
    SEQ_LEN = 10
    D_MODEL = 512
    D_FF = 2048
    
    # 1. Setup the Model
    model = feed_forward.PositionwiseFeedForward(D_MODEL, D_FF)
    
    # 2. Create Dummy Data
    rng = jax.random.PRNGKey(0)
    key1, key2 = rng.split(rng)
    
    # Input x: (Batch, Seq, d_model)
    x = jax.random.normal(key1, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    
    # 3. Initialize Weights
    variables = model.init(key2, x)
    
    # --- TEST 1: Internal Anatomy (Did it expand?) ---
    print("--- Test 1: Checking Internal Dimensions ---")
    params = variables['params']
    
    # We expect two Dense layers. 
    # Dense_0 should be (512 -> 2048)
    # Dense_1 should be (2048 -> 512)
    
    layer0_shape = params['Dense_0']['kernel'].shape
    layer1_shape = params['Dense_1']['kernel'].shape
    
    print(f"Layer 1 (Expand) Weights:   {layer0_shape}")
    print(f"Layer 2 (Compress) Weights: {layer1_shape}")
    
    assert layer0_shape == (D_MODEL, D_FF), f"❌ Expansion layer wrong size! Expected ({D_MODEL}, {D_FF})"
    assert layer1_shape == (D_FF, D_MODEL), f"❌ Compression layer wrong size! Expected ({D_FF}, {D_MODEL})"
    print("✅ Internal expansion/compression structure is correct.\n")

    # --- TEST 2: Forward Pass (Input/Output Consistency) ---
    print("--- Test 2: Forward Pass Shape Check ---")
    
    output = model.apply(variables, x)
    
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {output.shape}")
    
    assert output.shape == x.shape, "❌ Output shape does not match Input shape!"
    print("✅ Forward pass successful. Shapes align perfectly.")
    
    print("\n🎉 ALL FFN TESTS PASSED!")

if __name__ == "__main__":
    run_ffn_tests()