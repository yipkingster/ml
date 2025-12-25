
from jax import random
from jax import numpy as jnp
import jax.nn
import mlp


def main():
    key = random.PRNGKey(0)
    # Input: 28x28=784
    # Hidden Layer: MNIST standard 512
    # Output: 10 digit.
    layer_widths = [784, 512, 10]
    params = mlp.init_mlp_params(layer_widths, key)
    # --- Mock Data Setup for MNIST ---
    # 1. Fake MNIST Data (Batch of 5 images)
    #    Flattened 28x28 = 784 inputs
    dummy_images = random.normal(key, (5, 784))

    # 2. Fake Labels (Integers 0-9) for batch of 5
    dummy_labels_int = jnp.array([0, 5, 9, 1, 2])

    # 3. ONE-HOT ENCODE the labels in batch of 5
    # e.g. 5 becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    dummy_labels_onehot = jax.nn.one_hot(dummy_labels_int, num_classes=10)

    # Now run update with the one-hot labels
    new_params = mlp.update(params, dummy_images, dummy_labels_onehot, learning_rate=0.01)

    print("Did params change?", params[0]['w'][0,0] != new_params[0]['w'][0,0]) 
    # Should print True

if __name__ == "__main__":
    main()