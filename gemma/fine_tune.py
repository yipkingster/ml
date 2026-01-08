import dataclasses
import jax
import optax
from recurrentgemma import jax as recurrentgemma 
from flax.training import train_state
import jax.numpy as jnp
from recurrentgemma import common
import itertools
import loss_fn

# Module 1: Creating a config.
# @dataclasses.dataclass is a code generator that saves you time.
# It looks at the variables you declared in your class and automatically writes
# the "boring" standard methods for you (like __init__, __repr__, and __eq__) so
# you don't have to write them manually.
@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    # Model Hparams
    vocab_size: int = 256000
    seq_len: int = 512
    
    # Optimization Hparams
    learning_rate: float = 1e-4
    batch_size: int = 2  # Keep small for Colab
    num_steps: int = 100
    
    # Checkpoint Path (optional)
    ckpt_path: str = "path/to/pretrained/checkpoint"

# In JAX, since frozen is True, the object is immutable. If necessary, use
# replace() to copy and update the object.
# new_config = dataclasses.replace(config, val=2)
config = TrainingConfig()

# Module 2: Model Initialization & Train State
# In Flax, we don't just hold "weights" in a variable. We hold a TrainState
# object that bundles the Parameters (weights) + Optimizer State (momentum, etc.).

def _get2b_config(config):
    """Get default RecurrentGemma 2B config."""
    
    # RecurrentGemma-2B (26 layers) follows a repeating pattern of 2 Recurrent 
    # Layers followed by 1 Local Attention Layer.Pattern: Recurrent > Recurrent
    # > Attention ... (repeat).
    # Remainder: Since 26 isn't divisible by 3 (8 sets of 3 + 2 left over), it
    # ends with Recurrent > Recurrent.
    # Define the lay pattern (RECURRENT, RECURRENT, ATTENTION)
    pattern = (
        common.TemporalBlockType.RECURRENT,
        common.TemporalBlockType.RECURRENT,
        common.TemporalBlockType.ATTENTION
    )
    # Fill the list until we reach 26 layers
    target_depth = 26
    block_types = tuple(itertools.islice(itertools.cycle(pattern)), target_depth)

    # Instantiate config:
    griffin_config = recurrentgemma.GriffinConfig(
        vocab_size=config.vocab_size,
        width=2560,                      # d_model
        mlp_expanded_width=7680,         # 3 x width
        num_heads=10,                    # d_model 2560 / head_dimension 256
        block_types = block_types,       # Tuple generated above
        embeddings_scale_by_sqrt_dim=True,   # Standard Gemma scaling
        attention_window_size=2048,
        # This is a stability technique introduced in the Gemma 2 and
        # RecurrentGemma family. It prevents the output numbers (logits) from 
        # getting too large, which can cause the training loss to explode 
        # (NaNs). It essentially "caps" the confidence of the model so it never 
        # screams "I AM 100% SURE!" which helps training stability.
        logits_soft_cap=30.0,            # Required for RecurrentGemma-2B
        # The RG-LRU (Recurrent Gated Linear Recurrent Unit) is the
        # "secret sauce" of the Griffin architecture. It is the component that
        # allows RecurrentGemma to have a massive context length without running
        # out of memory.
        lru_width=2560,
        scan_type=common.ScanType.AUTO
    )

    return griffin_config

def create_train_state(rng, config):
    """Initializes the model, optimizer, and training state."""
    
    # 1. Setup Model Config
    # Load GriffinConfig.
    model_config = _get2b_config(config)
    
    # 2. Instantiate the Model
    model = recurrentgemma.Griffin(config=model_config)
    
    # 3. Initialize Parameters
    # You need a dummy input shape to trigger initialization.
    # RecurrentGemma usually needs: tokens, segment_pos
    dummy_tokens = jnp.zeros((1, config.seq_len), dtype=jnp.int32)
    # TODO: Run model.init(...) to get 'params'
    key = jax.random.key(0)
    params = model.init(key, dummy_tokens, train=True) 

    # 4. Define Optimizer
    # Use optax.adamw with the learning rate from config
    # See https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamw
    tx = optax.adamw(config.learning_rate)

    # 5. Create TrainState
    # This wrapper holds everything together.
    # Note: We pass 'model' as a static field so we can use it inside the step
    # function later.
    class TrainState(train_state.TrainState):
        model: recurrentgemma.Griffin = dataclasses.field(static=True, default=None)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        model=model # Storing the model instance itself
    )
    
    return state

@jax.jit
def train_step(state, batch, rng):
    """Updates the model parameters for a single step."""
    
    # 1. Handle RNG
    # Split the incoming rng key to get a fresh one for dropout
    dropout_rng, new_rng = jax.random.split(rng)
    
    # 2. Calculate Gradients
    # HINT: usage is value_and_grad(fun)(params, ...)
    # We defined loss_fn to take (params, state, ...), so we differentiate wrt arg 0
    # jax.value_and_grad runs forward pass and derivative in one run to save
    # compute resource, otherwise we need to run jax.grad(loss) and then run
    # f(x) to get the value, which runs forward pass twice.
    grad_fn = jax.value_and_grad(loss_fn.loss_fn, argnums=0)
    loss, grads = grad_fn(state.params, state, batch, dropout_rng)
    
    # 3. Update State
    # state.apply_gradients(grads=...) handles the optimizer update logic
    new_state = state.apply_gradients(grads=grads)
    
    # loss isn't necessary for training itself but we return it to:
    # 1. Serve as a dashboard to show how training is going.
    #   If Loss goes DOWN: The model is learning.
    #   If Loss stays FLAT: Something is broken (learning rate too low, bad data).
    #   If Loss goes UP: The model is exploding (learning rate too high).
    #   If Loss is NaN: You have a numerical error (precision issue).
    # 2. Adjust or stop the training early:
    #   ReduceLROnPlateau: "If the loss hasn't improved in 100 steps, cut the
    #   learning rate by half."
    #   Early Stopping: "If the loss hasn't improved in 1000 steps, stop
    #   training completely to save money."
    return new_state, loss, new_rng