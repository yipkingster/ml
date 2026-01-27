import jax
import jax.numpy as jnp

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Computes the attention scores and context vector.
    Note that this function only operates on the last 2 dimentsion of all q, k
    and v so it doesn't matter q, k or v have 3 or 4 dimensions. They just need
    to have the same shapes.
    
    Args:
        q: Queries. Shape (batch, seq_len, d_q) or (batch, head, seq_len, d_q)
           e.g., (Batch 32, Sequence 50 words, Dimension 64)
           Batch: 32 independent sentences to analyze.
           Sequence: Content Vector of the first 50 words of each sentence.
           Dimension: 64 floats as embedding for each word.
        k: Keys. Shape (batch, seq_len, d_k) or (batch, head, seq_len, d_k)
           Same batch but different embeddings to represent keys (semantic info,
           adj, noun, etc).
        v: Values. Shape (batch, seq_len, d_v) or (batch, head, seq_len, d_v)
           Same batch but different embeddings to represent values (content
           info, as "he" means man, male)
        mask: Optional boolean mask (batch, seq_len, seq_len) to hide padding.
        
    Returns:
        output: The context vector. Shape (batch, seq_len, d_v) or 
           (batch, head, seq_len, d_v)
           Context vector to hold "relatedness" between a word in the sentence 
           with all other words in the same sentence.
        attention_weights: The raw probability scores. Shape (batch, seq_len, 
        seq_len)
           This is for engineer's debugging purposes.
    """
    # 1. Calculate Raw Scores (The "Affinity/Similarity" matrix)
    k_t = jnp.swapaxes(k, -2, -1)
    similarity_mx = jnp.matmul(q, k_t)
    
    # 2. Scale the scores
    # Since we calculate the sum of all elements in the query vector, the values
    # are inflated by d_k, so variance would be increased by d_k^2. In order to
    # keep the variance consistent at 1 unit, we need to divide the value by
    # sqrt(d_k).
    # TODO: Go through "Prove mathmetically that by dividing the scores by 
    # sqrt(d_k), the variance is back to 1."
    d_k = q.shape[-1]
    scaled_s_mx = similarity_mx/jnp.sqrt(d_k)
    
    # 3. Apply Mask (Optional for now, but good to know)
    # If mask is provided, set masked positions to -1e9 before softmax. Note
    # that exp(-1e9) is approximately 0.
    # Those numbers before softmax applies exp() to them are called logits.
    large_negative = -1e9
    masked_mx = jnp.where(mask==True, scaled_s_mx, large_negative) if mask!=None else scaled_s_mx

    # 4. Softmax
    # Apply jax.nn.softmax. BE CAREFUL with the 'axis'.
    # We want the probabilities to sum to 1 across the *last* dimension (the keys).
    # Note the masked_mx has shape (batch, seq_len, seq_len). e.g. (32, 50, 50)
    attention_weights = jax.nn.softmax(masked_mx, -1)
 
    # 5. Weighted Sum
    # Multiply weights with V.
    return jnp.matmul(attention_weights, v), attention_weights
