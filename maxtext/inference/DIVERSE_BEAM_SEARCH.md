# Diverse Beam Search (DBS) Implementation in MaxText

This document outlines the design and implementation process of integrating Diverse Beam Search (DBS) into the MaxText inference framework.

## 1. Why Diverse Beam Search?
Currently, MaxText primarily supports greedy and probabilistic sampling strategies (Top-K, Nucleus). While these are excellent for "chat-style" applications that prioritize low-latency streaming to improve the user experience, they can sometimes lead to repetitive or low-quality generations for complex reasoning or long-form tasks.

**Diverse Beam Search (DBS)** aims to improve generation quality by exploring multiple parallel paths (beams) simultaneously. It incorporates a diversity penalty that discourages different beam groups from selecting the same tokens, resulting in a more varied and higher-quality set of candidates. 

**Note on Streaming**: Implementing DBS requires sacrificing real-time streaming capability. Because beams can be reordered at any step based on their cumulative scores, the "best" token for a given position might change as the sequence progresses. Therefore, sequences are only finalized once the entire generation process is complete.

## 2. The Mathematics of DBS

The implementation is based on the paper [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models (Vijayakumar et al., 2016)](https://arxiv.org/abs/1610.02424).

### Standard Beam Search Objective
Since seeking the best probability of a sequence over the entire space of possible sequences is computationally impossible, standard BS acts as a greedy heuristic optimization across $B$ parallel beams (the `num_decodes`). At each time step $t$, it attempts to find the set of next tokens across all beams $Y_{[t]}$ that locally maximizes the probabilities when extending the previous sequence step states:

$$ \mathbf{Y}_{[t]} = \arg\max_{y_1^{[t]}, \ldots, y_B^{[t]} \in \mathcal{V} \text{ s.t. } y_i^{[t]} \neq y_j^{[t]}} \sum_{b=1}^B \log P(y_b^{[t]} \mid Y_{b, [t-1]}, X) $$

Here is the piece by piece explanation:
1. $B$ is the number of active beams.
2. The superscript $[t]$ placed in brackets (as in $y^{[t]}$ or $\mathbf{Y}_{[t]}$) denotes the **time step** index of the sequence generation process. For example, $y_b^{[t]}$ represents the single token being evaluated exactly at time step $t$ for beam $b$.
3. The "s.t." in the term $\text{s.t. } y_i^{[t]} \neq y_j^{[t]}$ means "subject to". It simply forces the algorithm to select $B$ distinct candidate tokens in each beam, avoiding duplication at a single timestep. 
4. The summation run $\sum_{b=1}^B$ means the algorithm searches for a **set** of next-tokens that maximizes the probabilities across all $B$ concurrently active beam-branches strictly localized at step $t$.
5. $\log P$ ("Log-Probability"): The natural logarithm of the probability score assigned by the neural network.  We use the sum of log-probabilities instead of product of raw probabilities to avoid numerical underflow by multiplying many small numbers together.
6. ($y_b^{[t]} \mid Y_{b, [t-1]}, X$): Given the prompt/input $X$ (e.g. the translating sentence) and the previous context $Y_{b, [t-1]}$ leading up to this point within beam $b$, we find the specific next token $y_b^{[t]}$ that maximizes that beam's conditional probability.

### Diverse Beam Search Objective
DBS partitions the total beams ($B$) into groups ($G$). It optimizes these groups sequentially at each time step. The first group ($g=1$) acts like standard BS. For any subsequent group $g$, a penalty $\Delta$ is applied to discourage the selection of tokens that were already chosen by the previous groups $\{1, \ldots, g-1\}$ at that identical timestep $t$.

$$ \mathbf{Y}_{[t]}^{[g]} = \arg\max_{y_1^{[t]}, \ldots, y_{B'}^{[t]} \in \mathcal{V}} \sum_{b=1}^{B'} \left( \log P(y_b^{[t]} \mid Y_{b,[t-1]}^{[g]}, X) + \lambda \sum_{h=1}^{g-1} \Delta(y_b^{[t]}, y_{b}^{[t], [h]}) \right) $$

Here is the piece by piece explanation:
1. $\mathbf{Y}_{[t]}^{[g]}$ ("Y at step t for group g"): The set of $B'$ tokens chosen for group $g$ at time step $t$.
2. $\arg\max_{y_1^{[t]}, \ldots, y_{B'}^{[t]} \in \mathcal{V}}$: The $\arg\max$ function. It looks through the entire vocabulary ($\mathcal{V}$) to find the next-tokens that maximize the total score of the summation next to it.
3. $\sum_{b=1}^{B'}$: We sum the scores across all $B'$ parallel beam groups. (Where $B'$ is simply the total beams $B$ divided by the number of groups $G$).
4. $\log P(y_b^{[t]} \mid Y_{b,[t-1]}^{[g]}, X)$: The log-probability that the model's neural network assigns to the candidate token $y_b^{[t]}$, given the original prompt $X$ and the sequence selected before this timestep for this specific beam group $g$: $Y_{b,[t-1]}^{[g]}$.
5. $\lambda$: The `diversity_strength` penalty multiplier. It controls how severely we want to punish the model for duplicate tokens.
6. $\sum_{h=1}^{g-1}$: A loop that iterates through every single previous group ($h$) that has already been decided at this particular step $t$ (from group $1$ up to $g-1$).
7. $\lambda \sum \Delta(y_b^{[t]}, y_{b}^{[t], [h]})$: The **Hamming Diversity Penalty**. By multiplying the number of occurrences of this token by $\lambda$, we subtract a penalty proportional to how many times the exact same token has already been selected by older groups at this time step.

## 3. Design of the DBS Implementation
The integration required significant changes across four core areas of the MaxText engine:

### 3.1 Decode State Modification
To support multiple paths, the `decode_state` was expanded to handle the multiplied search space.
*   **Batch Expansion**: All state entries (tokens, next positions, generated tokens) were updated to have a leading dimension of `batch_size * num_beams`. This ensures that every beam has its own independent state.
*   **Score Tracking**: A `cumulative_logprobs` array was added to the state to store the running total of each path's log probabilities, allowing for global ranking of beams.
*   **Beam Synchronization**: An `is_dbs` flag was added to the state to allow the JIT-compiled engine to dynamically switch between standard and DBS sampling paths.

### 3.2 Sampling Function Signature
The sampling interface in `inference_utils.py` was overhauled to support the complex requirements of DBS:
*   **State-Aware Sampling**: Unlike standard sampling, DBS requires access to the `cumulative_logprobs` of the parents and the group configuration (`num_groups`, `diversity_penalty`).
*   **Multi-Winner Return**: The sampling function now returns three critical arrays: the **New Tokens**, the **Updated Scores**, and the **Parent Indices**. The latter is essential for identifying which parent beam produces each successful candidate.

### 3.3 Memory Management: KV Cache Reordering
This is the most technically complex part of the implementation. When a beam is selected as a winner for the next step, its entire history (the Key-Value cache) must follow it.
*   **Reorder Logic**: We implemented `MaxEngine.reorder_cache`, which uses `jax.tree_util.tree_map` and `jnp.take(old_cache, parent_indices, axis=0)`.
*   **Memory Swapping**: This effectively "swings" the model's memory so that successful parents overwrite the memory slots of killed or lower-performing beams.

### 3.4 User Interface: `decode.py` and Streaming
The end-user interface was updated to handle the transition from streaming to batch-finalized output:
*   **Streaming Bypass**: When `decode_sampling_strategy="diverse_beam_search"` is selected, the engine buffers tokens on the host (CPU) rather than immediately piping them to the output.
*   **Final Decoding**: Results are only detokenized and printed once the EOS (End of Sequence) is reached or the maximum length is exceeded.

## 4. Challenges and Resolutions
During the implementation, several critical hurdles were overcome:

*   **Redundant Scaling (Double-Multiplication)**: We initially encountered shape mismatches (e.g., 16 vs 4) because the batch size was being scaled both during engine initialization and again inside the AOT compilation logic. Resolve: We moved batch scaling to a centralized `SimpleNamespace` proxy configuration used during engine setup.
*   **ZeroDimension Errors**: The sampling logic crashed when `top_k` was set to 0 (the default). Resolve: Updated `inference_utils.py` to correctly handle `pool_size` defaults when `top_k` is zero.
*   **JIT Tracing Collisions (`jax.lax.cond`)**: JAX's `cond` traces both branches, leading to shape errors when the DBS branch (Batch 4) was compared to the Standard branch (Batch 1). Resolve: Replaced `jax.lax.cond` with a static Python `if` statement based on a `dbs_requested` flag, ensuring only the correctly-sized path is compiled for a given configuration.
*   **Tracer Conversion Failures**: We initially tried to use the traced `is_dbs` array in Python `if` blocks, which is disallowed in JIT. Resolve: Switched to the static `dbs_requested` boolean derived from the immutable configuration.

## 5. Limitations and Constraints
The current implementation of DBS has several important constraints:

*   **Streaming Disabled**: Real-time streaming is intentionally disabled for DBS to avoid "streaming jitter" where the printed output might change as different beams become the global winner.
*   **No Runtime Switching**: We have intentionally avoided supporting run-time switching of the sampling algorithm (e.g., from Greedy to DBS or vice versa). The JAX JIT-compiled functions are statically branched based on the initial engine configuration to reduce code complexity and improve stability. Users are encouraged to maintain separate engine instances if multiple sampling strategies are needed simultaneously.
*   **Relaxed Dependency Checks**: Dependency checks (like JetStream) for certain packing features are disabled when `DECOUPLE_GCLOUD=TRUE`. This prevents local CPU testing from failing unnecessarily when these components are not actually being used.
