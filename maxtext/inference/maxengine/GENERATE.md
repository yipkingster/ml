# Generate Function Explained

The **[generate()](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:944:2-980:64)** function is the "Engine of Thought." It represents a single loop of the model typing out one word of its response. 

Everything in this function is designed to be **Autoregressive**, meaning it uses the word it *just* typed to decide what to type next.

---

### 1. Input Parameters
Because this function is JIT-compiled for speed, it expects highly specific shapes:

| Parameter | Meaning | Typical Shape |
| :--- | :--- | :--- |
| **[params](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:218:2-282:17)** | The model's weights. | (Dictionary of Tensors) |
| **[decode_state](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1599:2-1703:17)** | The **"Context Bag."** Contains the KV Cache and counters. | (Dictionary) |
| **`decode_state['tokens']`** | The token we generated in the **previous** step. | [(batch, 1)](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/decode.py:89:0-310:21) |
| **`decode_state['next_pos']`** | The 3D coordinates for the word we are about to pick. | [(batch, 1)](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/decode.py:89:0-310:21) |
| **[page_state](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/page_manager.py:64:0-97:3)** | The Memory Map (for Paged Attention). | [(PageState Object)](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/decode.py:89:0-310:21) |
| **`rng`** | Random seed (for non-greedy sampling). | `PRNGKey` |
| **`algorithm`** etc. | Settings like `temperature` or [topk](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/inference_utils.py:164:0-171:23). | (Scalars) |

---

### 2. Internal Action: The Autoregressive Step
Inside the function, the model performs these operations in order:
1.  **The Forward Pass**: It sends the *previous token* through the network. It lookups the *KV Cache* (the past) to understand the context.
2.  **Logit Production**: It creates a probability map for the entire vocabulary (e.g., 32,000 words). Shape: **[(batch, 1, vocab_size)](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/decode.py:89:0-310:21)**.
3.  **Sampling**: It uses your settings (like "Greedy" or "Nucleus") to pick exactly **one token** from that map.

---

### 3. Output Values
The function returns a tuple containing the **Brand New State** and the **Result**.

#### A. The Updated `DecodeState` (Ready for the *next* loop)
*   **[tokens](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/multimodal/processor_qwen3_omni.py:509:0-640:45)**: The brand new word we just picked. Shape: **[(batch, 1)](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/decode.py:89:0-310:21)**.
*   **[cache](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:888:4-930:15)**: The **Updated KV Cache**. It now includes the "memory" of the word it just picked. 
*   **`next_pos`**: Incremented by 1 (e.g., position 10 becomes 11).
*   **`generated_tokens`**: Incremented by 1.

#### B. The `ResultTokens` Object (The "Answer Page")
This is what actually gets returned to your screen.
*   **Shape**: [(batch, 3)](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/decode.py:89:0-310:21)
*   **Contents**:
    *   ID of the new word.
    *   Validity (1 if the word is real).
    *   The total length of the answer so far.

---

### The Big Picture:
If you want a 50-word answer, the system calls [generate()](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:944:2-980:64) **50 times in a row**, each time feeding the "Output" of step 1 as the "Input" to step 2. This cycle continues until the model picks a "Stop" token (like an EOS or a period).

**In summary:** [generate()](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:944:2-980:64) is a **Single Step Transformer**. It takes the **Current Context** and returns a **Single Word + New Context**.

Created PAGE_STATE.md

# Sampler
Searched for "sampler("
Viewed inference_utils.py:150-171

The **"Sampler"** is the model's **Decision Maker.** It is the final step in the model's brain where it picks one specific word from a list of thousands of possibilities.

### 1. The Raw Input (Logits)
Before the sampler runs, the model produces **Logits**.
These are a list of raw numbers for every word in the dictionary (e.g., 32,000 words). High numbers mean "This is a likely word," and low numbers mean "This makes no sense."

### 2. The Sampler's Job
The sampler takes those probabilities and picks exactly **one token.** 

MaxText supports several "strategies" for picking that token:
*   **Greedy Sampling**: Always pick the word with the absolute highest score. (Predictable, good for math or code).
*   **Top-K Sampling**: Look only at the top 50 words. Ignore the thousands of unlikely ones. This prevents the model from saying something nonsensical.
*   **Nucleus (Top-P) Sampling**: Look at the smallest group of words that together make up 90% of the probability. This is the most common for natural chat.
*   **Temperature**: A scaling factor. Low temperature (e.g., 0.1) makes the sampler confident and boring; high temperature (e.g., 1.5) makes it "random" and creative.

### 3. As a Parameter in [MaxEngine](cci:2://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:99:0-1737:29)
In your [MaxEngine](cci:2://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:99:0-1737:29) code, you see **`sampler: Callable`**. This is an optional hook that allows you to:
1.  **Use the Default**: If you leave it blank, the engine uses **`inference_utils.sampling`**, which handles all the common strategies above automatically.
2.  **Provide a Custom One**: If you are a researcher building a new way to pick words (like Beam Search or Contrastive Search), you can pass your own custom function as the `sampler`, and the engine will use your logic instead.

**In summary:** The model does the math, but the **Sampler** makes the choice. It turns "This word is 85% likely" into "Actually type out the word 'Hello'."
