import jax
import jax.numpy as jnp
from encoder_layer import EncoderLayer 

def run_tests():
    print("🧪 Running Encoder Layer Tests...\n")

    # --- Configuration ---
    BATCH_SIZE = 2
    SEQ_LEN = 10
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    DROPOUT_RATE = 0.1
    
    # 1. Setup Data
    rng = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(rng, 3)
    
    # Input: (Batch, Seq, d_model)
    x = jax.random.normal(key1, (BATCH_SIZE, SEQ_LEN, D_MODEL))
    
    # Mask: (Batch, 1, 1, Seq) - Standard attention mask shape
    # 1 means "attend", 0 means "ignore"
    mask = jnp.ones((BATCH_SIZE, 1, 1, SEQ_LEN))
    
    # 2. Initialize Model
    model = EncoderLayer(
        d_model=D_MODEL, 
        num_heads=NUM_HEADS, 
        d_ff=D_FF, 
        dropout_rate=DROPOUT_RATE
    )
    
    # Init needs a PRNG key for dropout and initialization
    print("... Initializing Weights")
    variables = model.init(key2, x, mask, training=False)
    
    # --- TEST 1: Shape Integrity ---
    print("\n--- Test 1: Shape Check ---")
    output = model.apply(variables, x, mask, training=False)
    
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {output.shape}")
    
    assert output.shape == x.shape, "❌ Output shape must match input shape!"
    print("✅ Shape preserved. This layer can be stacked!")

    # --- TEST 2: Dropout Toggle ---
    print("\n--- Test 2: Dropout (Training vs Eval) ---")
    
    # Run in EVAL mode (training=False) twice
    # Should be identical (Deterministic)
    out_eval_1 = model.apply(variables, x, mask, training=False)
    out_eval_2 = model.apply(variables, x, mask, training=False)
    
    # Run in TRAIN mode (training=True) twice
    # Should be different (Stochastic) because dropout removes random neurons
    # We need a 'dropout' RNG key for this
    out_train_1 = model.apply(variables, x, mask, training=True, rngs={'dropout': key3})
    out_train_2 = model.apply(variables, x, mask, training=True, rngs={'dropout': jax.random.fold_in(key3, 1)})
    
    # Check Eval Consistency
    diff_eval = jnp.sum(jnp.abs(out_eval_1 - out_eval_2))
    assert diff_eval == 0.0, "❌ Eval mode should be deterministic!"
    print(f"✅ Eval mode is deterministic (Diff: {diff_eval})")
    
    # Check Train Randomness
    diff_train = jnp.sum(jnp.abs(out_train_1 - out_train_2))
    assert diff_train > 0.0, "❌ Training mode should be random (Dropout didn't trigger)!"
    print(f"✅ Training mode is random (Diff: {diff_train:.4f})")
    
    print("\n🎉 ALL ENCODER LAYER TESTS PASSED!")

if __name__ == "__main__":
    run_tests()