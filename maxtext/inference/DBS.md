# Design of Diverse Beam Search (DBS)
In brief:
1. Update decode_state to include necessary state for DBS.
2. Change the signature of sampling to make it return all N beams with M groups and their .
3. In the generate function, update the cache and return the current winner while maintaing the beam search state in decode_state.
4. decode.py will not do "streaming" anymore. It will just print the final result.

This is a **Comprehensive Implementation Plan** for Diverse Beam Search (DBS). I have merged your original skeleton with our technical deep dives on JAX memory, the 16-to-4 search space, and the "Streaming Paradox."

### Phase 1: State Expansion (The "Memory Blueprint")
You will modify the [init_decode_state](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1607:2-1711:17) in the **[MaxEngine](cci:2://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:99:0-1745:29)** to handle the multiplied search space.

1.  **Multiply the Batch**: Every state entry (tokens, next_pos, generated_tokens) must now have a leading dimension of **`batch_size * num_beams`**.
2.  **Add Cumulative Scores**: Add a `cumulative_logprobs` array to the [decode_state](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1607:2-1711:17) dictionary. This stores the running total of a path’s score, not just the last word.
3.  **Add Parent Tracking**: Add a `parent_indices` placeholder to the state. 

### Phase 2: The Search Strategy (The "Math Wrapper")
You will update **[inference_utils.py](cci:7://file:///Users/kevinwang/Projects/ml/maxtext/inference/inference_utils.py:0:0-0:0)** to perform the Diverse Search math.

1.  **DBS Sampler**: Implement `sample_diverse_beam_search`.
    *   **Expand (16 Candidates)**: For each of the $k$ parent beams, find the Top $k$ most likely next words. 
    *   **Diverse Penalty**: Divide the $k$ beams into $M$ groups. Subtract the `diversity_strength` from the logits of any word that a previous group has already selected.
    *   **Choose (The Global 4)**: Out of the 16 penalized candidates, pick the top 4 winners overall across all groups.
2.  **Return Data**: Return both the **New Token IDs** and the **Parent Indices** (which parent beam "gave birth" to each of the 4 winners).

### Phase 3: The Engine Loop (The "Memory Orchestrator")
You will modify the **`MaxEngine._generate_jit`** method to handle the path-switching.

1.  **Run the Sampler**: Call your new DBS function to get the winners and their parents.
2.  **Re-order the KV Cache (Crucial)**: Use `jax.tree_util.tree_map` and `jnp.take(old_cache, parent_indices, axis=0)`. 
    *   This "swings" the model's memory so that successful parents overwrite the memories of the unsuccessful/cilled beams.
3.  **The "Current Best" Picker**: At the very end of the JIT function, find the single beam with the highest `cumulative_logprob`. 
4.  **The Shipping Invoice**: Populate the **`ResultTokens`** object with **ONLY that single winner's word**. 

### Phase 4: User Interface ([decode.py](cci:7://file:///Users/kevinwang/Projects/ml/maxtext/inference/decode.py:0:0-0:0))
You will make small configuration changes to ignite the new engine.

1.  **The Flag**: Add `--sampling_strategy="diverse_beam_search"` and `--num_beams=4` to your CLI arguments.
2.  **The Silent Print**: Because the engine only returns the current best winner in the result object, [decode.py](cci:7://file:///Users/kevinwang/Projects/ml/maxtext/inference/decode.py:0:0-0:0) remains "simple" and prints a single stream of text. 
    *   **Note**: As discussed, the printed text may "jitter" slightly as the model changes its mind between beams. For perfectly stable output, you can turn off streaming and just print the final `sampled_tokens_list` at the very end.

---

### In Summary:
1.  **[init_decode_state](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1607:2-1711:17)**: Prepares the space for Beams.
2.  **[sampling](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/inference_utils.py:124:0-146:61)**: Does the "16-to-4" math with diversity penalties.
3.  **[_generate_jit](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:982:2-1093:13)**: Physically **moves the model's memory** to follow the winners.
4.  **`ResultTokens`**: Distills all the complexity back down to **1 word** for [decode.py](cci:7://file:///Users/kevinwang/Projects/ml/maxtext/inference/decode.py:0:0-0:0) to print.

**See REORDER_CACHE.md for the design of reorder_cache.**
