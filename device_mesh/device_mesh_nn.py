import os

# Set flag to simulate 8 devices
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=8"

# jax must be imported after setting the flag
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

def predict(params, inputs):
    # inputs is a batch with shape batch x d_model
    # params is a matrix with shape (n-layer - 1) x layer_size x next_layer_size 
    for W, b in params:
        # W is a matrix with shape d_model x d_model (Each node has d_model
        # Weights for the next d_model nodes)
        # Inputs is a matric with shape batch x d_model
        # The dot product is of shape: batch x d_model
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.maximum(outputs, 0)
    return outputs

# Calculate loss after 1 pass of the entire batch data.
@jax.jit
def loss(params, batch):
    inputs, targets = batch
    outputs = predict(params, inputs)
    # outputs has shape of batch x d_model (axis=-1)
    # targets has shape of 1 x d_model
    loss = jnp.mean(jnp.sum((outputs - targets)**2, axis=-1))
    return loss

def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    # Xavier init.
    # This brings the variance of the output comparable with the variance of 
    # input.
    W = jax.random.normal(key, (n_in, n_out))/jnp.sqrt(n_out)
    b = jnp.zeros((n_out,))
    # Returns a tuple.
    return W, b

def init_model(key, layer_sizes, batch_size):
    '''Initialize the model.
    Args:
        key: init key
        layer_sizes: an array including # of nodes in each layer.
        batch_size: Supported size in batching.
    Returns:
        params, (input, targets)
    '''
    key, *keys = jax.random.split(key, len(layer_sizes))
    # Creating W list from i to i+1 (i : [0, layer_sizes-2])
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

    inputs = jax.random.normal(key, (batch_size, layer_sizes[0]))
    targets = jax.random.normal(key, (batch_size, layer_sizes[-1]))

    return params, (inputs, targets)

def main():
    layer_sizes = [784, 8192, 8192, 8192, 10]
    batch_size = 8192
    params, batch = init_model(jax.random.key(0), layer_sizes, batch_size)
    # New version:
    # mesh = jax.make_mesh((8,), ("batch",))
    devices = mesh_utils.create_device_mesh((8,))
    mesh = Mesh(devices, axis_names=("batch",))
    batch_sharding = NamedSharding(mesh, P("batch"))
    rep_sharding = NamedSharding(mesh, P())
    batch = jax.device_put(batch, batch_sharding)
    params = jax.device_put(params, rep_sharding)

    print("Intial Loss:")
    print(loss(params, batch))

    gradfun = jax.jit(jax.grad(loss))

    step_size = 1e-5
    for _ in range(5):
        grads = gradfun(params, batch)
        params = [(W - step_size*dW, b - step_size*db)
                  for (W, b), (dW, db) in zip(params, grads)]
    print("Final Loss:")
    print(loss(params, batch))

if __name__ == "__main__":
    main()