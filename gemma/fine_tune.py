import dataclasses
from recurrentgemma import jax as recurrentgemma 
from flax.training import train_state
import jax.numpy as jnp

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

def create_train_state(rng, config):
    """Initializes the model, optimizer, and training state."""
    
    # 1. Setup Model Config
    # HINT: Load GriffinConfig, then use dataclasses.replace to fix attention_window_size
    # Expand the following line to ...GriffinConfig(vocab_size=config.vocab_size, ...)
    model_config = recurrentgemma.GriffinConfig(vocab_size=config.vocab_size)
    # TODO: Fix window size here
    
    # 2. Instantiate the Model
    model = recurrentgemma.Griffin(config=model_config)
    
    # 3. Initialize Parameters
    # HINT: You need a dummy input shape to trigger initialization.
    # RecurrentGemma usually needs: tokens, segment_pos
    dummy_tokens = jnp.zeros((1, config.seq_len), dtype=jnp.int32)
    # TODO: Run model.init(...) to get 'params'
    params = ... 

    # 4. Define Optimizer
    # HINT: Use optax.adamw with the learning rate from config
    tx = ... 

    # 5. Create TrainState
    # This wrapper holds everything together.
    # Note: We pass 'model' as a static field so we can use it inside the step function later.
    class TrainState(train_state.TrainState):
        model: recurrentgemma.Griffin = dataclasses.field(static=True, default=None)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        model=model # Storing the model instance itself
    )
    
    return state