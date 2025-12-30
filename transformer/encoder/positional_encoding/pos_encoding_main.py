import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pos_encoding as pe

def run_tests():
    # Configuration
    BATCH_SIZE = 2
    SEQ_LEN = 50
    D_MODEL = 128  # Smaller d_model for easier visualization
    rng = jax.random.PRNGKey(0)

    # 1. Initialize Model
    model = pe.PositionalEncoding(d_model=D_MODEL, max_len=100)
    dummy_input = jnp.zeros((BATCH_SIZE, SEQ_LEN, D_MODEL))
    
    # Run init (this triggers setup -> init_pe)
    variables = model.init(rng, dummy_input)
    
    # 2. Extract the calculated PE matrix from the variables
    # Structure is usually {'consts': {'pe': ...}} or {'const': ...} depending on your name
    pe_matrix = variables['const']['pe'] # Shape (1, 100, 128)
    
    print(f"PE Matrix Shape: {pe_matrix.shape}")

    # --- TEST A: Shape Verification ---
    assert pe_matrix.shape == (1, 100, D_MODEL), f"Expected (1, 100, {D_MODEL}), got {pe_matrix.shape}"
    print("✅ Shape verification passed.")

    # --- TEST B: Math Verification ---
    # At position 0, arguments are 0.
    # index 0 (even) -> sin(0) = 0.0
    # index 1 (odd)  -> cos(0) = 1.0
    val_0 = pe_matrix[0, 0, 0]
    val_1 = pe_matrix[0, 0, 1]
    
    assert jnp.isclose(val_0, 0.0), f"Expected PE[0,0] to be sin(0)=0, got {val_0}"
    assert jnp.isclose(val_1, 1.0), f"Expected PE[0,1] to be cos(0)=1, got {val_1}"
    print("✅ Math (sin/cos at pos 0) passed.")

    # --- TEST C: Forward Pass ---
    # Ensure it adds correctly to input
    output = model.apply(variables, dummy_input)
    assert output.shape == dummy_input.shape
    # Since input was zeros, output should equal the sliced PE
    expected_slice = pe_matrix[:, :SEQ_LEN, :]
    assert jnp.allclose(output, expected_slice)
    print("✅ Forward pass broadcasting passed.")

    # --- TEST D: Visualization ---
    print("Generating Heatmap...")
    # Squeeze batch dim -> (100, 128)
    viz_data = np.array(pe_matrix[0]) 
    
    plt.figure(figsize=(10, 6))
    plt.imshow(viz_data, cmap='RdBu', aspect='auto')
    plt.title("Positional Encoding Matrix")
    plt.xlabel("Embedding Dimension (d_model)")
    plt.ylabel("Sequence Position")
    plt.colorbar(label="Value")
    plt.show()

if __name__ == "__main__":
    run_tests()