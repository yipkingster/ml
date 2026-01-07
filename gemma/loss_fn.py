import optax
import jax.numpy as jnp

def loss_fn(params, state, batch, dropout_rng):
    """Computes the loss for a single batch.
    Args:
        params:
            model weights, bias and embeddings. This must be there first arg
            because this is what jax.grad() use to run derivative against.
        state:
            This contains model logic. Unlike PyTorch, the model is separated
            from the params because the framework will call state.model.apply to
            run forward.
        batch:
            Inputs and targets. Contains tokens and segment_pos for forward
            pass, and targets needed by loss function.
        dropout_rng:
            Random number needed to be passed in for dropout rate. Since Jax
            function needs to be pure function we can't have it generate random
            numbers internally.
    """
    
    # Unpack batch
    inputs = batch['inputs'] # Shape: (B, T)
    targets = batch['targets'] # Shape: (B, T) - usually inputs shifted by 1
    
    # 1. Forward Pass
    # HINT: model.apply requires {'params': params}
    # You might need to generate a mask or segment_pos if your dataset doesn't provide it.
    batch_size, seq_len = inputs.shape
    segment_pos = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32),
        (batch_size, seq_len)
    )
    logits, _ = state.model.apply(
        {'params': params},
        tokens=inputs,
        segment_pos=segment_pos,
        rngs={'dropout': dropout_rng}
    )
    
    # 2. Shift for Next-Token Prediction
    # We predict t+1 given t.
    # Logits shape: (B, T, Vocab) -> Cut off the last step
    # Targets shape: (B, T)       -> Cut off the first step (if not already shifted in dataset)
    
    # The last dimension is the logits (unnormalized log prob) of all tokens in
    # the vocabulary.
    shift_logits = logits[:, :-1, :] 
    shift_targets = targets[:, 1:]   
    
    # 3. Calculate Loss
    # Here integer labels are used as opposed to wasteful one-hot encoding.
    loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_targets)
    
    # Average the loss over the sequence and batch
    return jnp.mean(loss)