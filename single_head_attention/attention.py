import jax
import jax.numpy as jnp

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Computes the attention scores and context vector.
    
    Args:
        q: Queries. Shape (batch, seq_len, d_q)
           e.g., (Batch 32, Sequence 50 words, Dimension 64)
           Batch: 32 independent sentences to analyze.
           Sequence: Content Vector of the first 50 words of each sentence.
           Dimension: 64 floats as embedding for each word.
        k: Keys. Shape (batch, seq_len, d_k)
           Same batch but different embeddings to represent keys (semantic info, adj, noun, etc).
        v: Values. Shape (batch, seq_len, d_v)
           Same batch but different embeddings to represent values (content info, as "he" means man, male)
        mask: Optional boolean mask (batch, seq_len, seq_len) to hide padding.
        
    Returns:
        output: The context vector. Shape (batch, seq_len, d_v)
           Context vector to hold "relatedness" between a word in the sentence with all other words in the same sentence.
        attention_weights: The raw probability scores. Shape (batch, seq_len, seq_len)
           This is for engineer's debugging purposes.
    """
    # 1. Calculate Raw Scores (The "Affinity/Similarity" matrix)
    k_t = jnp.transpose(k, (0,2,1))
    similarity_mx = jnp.matmul(q, k_t)
    
    # 2. Scale the scores
    # Since we calculate the sum of all elements in the query vector, the values are inflated by d_k, so variance
    # would be increased by d_k^2. In order to keep the variance consistent at 1 unit, we need to divide the value
    # by sqrt(d_k).
    d_k = q.shape[-1]
    scaled_s_mx = similarity_mx/jnp.sqrt(d_k)
    
    # 3. Apply Mask (Optional for now, but good to know)
    #    If mask is provided, set masked positions to -1e9 before softmax. Note that exp(-1e9) is approximately 0.
    large_negative = -1e9
    masked_mx = jnp.where(mask==True, scaled_s_mx, large_negative) if mask!=None else scaled_s_mx

    # 4. Softmax
    #    Apply jax.nn.softmax. BE CAREFUL with the 'axis'.
    #    We want the probabilities to sum to 1 across the *last* dimension (the keys).
    attention_weights = jax.nn.softmax(masked_mx, 2)
 
    # 5. Weighted Sum
    #    Multiply weights with V.
    return jnp.matmul(attention_weights, v), attention_weights
