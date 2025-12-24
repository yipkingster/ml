import jax.numpy as jnp
import jax
from jax import grad, jit, random

# 1. Initialization
def init_mlp_params(layer_widths, key):
    """
    Creates a list of parameters (weights and biases) for the network.
    
    Args:
        layer_widths: Number of neurons in each layer, e.g., [20, 40, 10]
        key: A jax.Array
        
    Returns:
        Weight and bias matrix applied to each layer. e.g. if layer 1 has 20 nodes and layer 2 has 40 nodes, the w matrix will have
        shape (20, 40); the b matrix will have shape (40,).
        Example structure:
        [
          {'w': array_layer1, 'b': array_layer1},
          {'w': array_layer2, 'b': array_layer2},
          ...
        ]
    """
    output = []
    newkey = key
    for i in range(len(layer_widths)-1):
        newkey, subkey = random.split(newkey)
        current_width = layer_widths[i]
        next_width = layer_widths[i+1]
        arr = random.normal(subkey, (current_width, next_width))
        # Xavier initialization
        weights = arr * jnp.sqrt(2.0/(current_width + next_width))
        biases = jnp.zeros((next_width,))
        output.append({'w': weights, 'b':biases})
    return output
        
# 2. Forward Pass
def forward(params, x):
    """
    Computes the output of the network.
    
    Args:
        params: The list of weights/biases returned by init_mlp_params
        x: Input data matrix of shape (batch_size, input_dim)
        
    Returns:
        The activations of the final layer, of shape (batch_size, last_layer_dimension)
    """
    layer_input = x
    for layer in params[:-1]:
        layer_output = jax.nn.relu(jnp.dot(layer_input, layer['w']) + layer['b'])
        layer_input = layer_output
    
    # Last layer without ReLU
    output = jnp.dot(layer_input, params[-1]['w']) + params[-1]['b']
    return output


# 3. Loss Function
def mse_loss(params, x, y_true):
    """
    Computes Mean Squared Error.
    
    Args:
        params: The model parameters (needed to pass to forward())
        x: Input data
        y_true: The actual target values
        
    Returns:
        A scalar float (the loss).
    """
    net_output = forward(params, x)
    return jnp.mean((net_output - y_true)**2)

# 4. Update Step (The "Training Loop" iteration)
@jit
def update(params, x, y_true, learning_rate):
    """
    Performs one step of Gradient Descent.
    new_params = params - d_Loss/d_Weight * learning_rate.
    
    Args:
        params: Current weights
        x: Input data
        y_true: Targets
        learning_rate: Float
        
    Returns:
        new_params: The updated weights after one step of descent.
    """
    # Can't do the following because params contains list and tuples that don't support math operators.
    # return params - jax.grad(mse_loss)(params, x, y_true) * learning_rate
    gradient = grad(mse_loss)(params, x, y_true)

    def leaf_func(leaf, dleaf):
        return leaf - dleaf * learning_rate

    new_params = jax.tree.map(leaf_func, params, gradient)
    return new_params
