# What is Value Head?

A **Value Head** is a small "add-on" to the LLM's architecture that allows it to predict a single number (a score) rather than a sequence of text. 

If the LLM's standard "head" is a **Language Modeling Head** (predicting the next token), the Value Head is a **Critic Head** (predicting the quality of the state).

### 1. Architectural View
In a standard Transformer, the final hidden state of the sequence is passed through a Linear layer that outputs a vector of size `[Vocab_Size]` (logits).

With a **Value Head**, you take that same hidden state and pass it through a different, very small Linear layer that outputs a single number: **`[1]`**.

*   **Hidden State** $\to$ **LM Head** $\to$ "What word comes next?"
*   **Hidden State** $\to$ **Value Head** $\to$ "How good is this reasoning so far (0.0 to 1.0)?"

---

### 2. Why it is useful for MCTS
In your JAX-MCTS project, the Value Head is your "shortcut" for evaluating nodes.
*   **Without a Value Head:** You have to ask the LLM to critique itself using a prompt (slow, costs tokens).
*   **With a Value Head:** Every time you generate a token or a thought, the model automatically produces a "confidence score" or "win probability" in the background.

---

### 3. The Actor-Critic Analogy
This concept comes from Actor-Critic Reinforcement Learning (like PPO):
*   **The Actor (LM Head):** The part of the brain that generates the thoughts and moves.
*   **The Critic (Value Head):** The part of the brain that stands back and says, "That move seems like it puts us in a winning position."

### 4. Implementation in MaxText
If you were to add a Value Head to MaxText, you would:
1.  Modify the model definition in `models.py`.
2.  Add a small `Dense(1)` layer at the output of the Transformer blocks.
3.  Train it on a dataset where you know which reasoning steps were correct and which were wrong.

**Note:** If your current model (like a base Llama 3 or Gemma 2) doesn't have a Value Head pre-trained, you will likely need to use **Self-Evaluation Prompts** or a separate **Reward Model** (like an Llama-Guard or a specialized Verifier) to get your MCTS scores.

# How are Different Head/Model Served?

If you use a separate model for evaluation (often called a **Reward Model** or **Outcome Verifier**), it functions as a "judge" that sits outside your main generation loop.

Here are the three ways this is typically architected in a high-performance environment like MaxText:

### 1. The "Co-Resident" Pattern (Shared VRAM)
Both your **Generator Model** and your **Reward Model** are loaded into memory on the same TPU/GPU cluster.
*   **Where it sits:** In your code, you would have two instances of `MaxEngine`: one for generating thoughts and one for scoring them.
*   **How it's invoked:** 
    1.  Generator produces a thought.
    2.  The thought string is passed to the Reward Model's `prefill()` or `apply()` function.
    3.  The Reward Model returns a scalar value.
*   **Pros:** Very low latency (no network calls).
*   **Cons:** High memory usage (you are storing two models).

---

### 2. The "Remote Verifier" Pattern (Microservice)
The Reward Model is hosted on a separate machine or a different TPU pod as a standalone service.
*   **Where it sits:** It is an external API endpoint (like a JetStream or vLLM server).
*   **How it's invoked:** 
    1.  Generator finishes a thought.
    2.  Your MCTS loop makes an asynchronous HTTP/gRPC call to the remote service, sending the full prompt + the new thought.
    3.  The service responds with a score.
*   **Pros:** Scales independently; the Generator has all the local VRAM.
*   **Cons:** Network latency adds up quickly if you are doing thousands of MCTS simulations.

---

### 3. The "Lightweight Verifier" (Parameter Sharing)
This is an advanced JAX optimization. Some Reward Models share the "backbone" (the weights) of the Generator but have a different "head."
*   **Where it sits:** Integrated into the same model definition.
*   **How it's invoked:** 
    In the same forward pass that predicts the next token, the model also outputs a "value" logit. This is theoretically the fastest method but requires a specialized model (like Llama-3-Instruct-Value).

---

### Workflow inside your MCTS Loop
Regardless of where the model sits, the invocation happens during the **Evaluation** step of your loop:

```python
# MCTS ITERATION
selected_node = select_node(tree)
thought = generate_thought(selected_node) # INVOKE GENERATOR (MaxEngine 1)

# EVALUATION STEP - This is where the separate model comes in
score = reward_service.get_score(prompt + thought) # INVOKE REWARD MODEL (MaxEngine 2 or API)

# BACKPROPAGATION
update_tree_stats(selected_node, score)
```

### Which one should you pick?
For your **MaxText** project:
1.  **If you have a massive TPU pod:** Try the **Co-Resident** pattern (two `MaxEngine` instances). It’s the easiest to debug locally.
2.  **If you are constrained on memory:** Use **Self-Evaluation**. Instead of a separate model, you just send a new prompt to the *same* `MaxEngine` that says: *"On a scale of 1-10, how consistent is the following thought? Output only the number."* This is what the official "Tree of Thoughts" paper does!
