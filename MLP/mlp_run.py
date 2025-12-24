
from jax import random
import mlp


def main():
    # Setup
    key = random.PRNGKey(0)
    layer_widths = [10, 20, 1] # Input 10 dim, Hidden 20 dim, Output 1 scalar
    params = mlp.init_mlp_params(layer_widths, key)

    # Dummy Data
    dummy_x = random.normal(key, (5, 10))  # Batch of 5, 10 features each
    dummy_y = random.normal(key, (5, 1))   # Batch of 5, 1 target each

    # Run one step
    new_params = mlp.update(params, dummy_x, dummy_y, 0.01)

    print("Did params change?", params[0]['w'][0,0] != new_params[0]['w'][0,0]) 
    # Should print True

if __name__ == "__main__":
    main()