# Refined Design: JAX-based Tree of Thoughts with Paged Attention

## Question

Tell me if my following thoughts on the right track:

Current maxengine calls prefill phase, then generate phase.
In the new Jax-MCTS, the prefill phase will not be touched. However, the generate phase will be wrapped in a MCTS loop:

The MCTS loop will create thought "nodes" initially by picking top K logits. Then run the MCTS loop while managing KV cache: 
1. If the generated token is not EOT - "End of Thought", continue call generate() to get the next token and update the KV cache.
2. If the generated token is EOT, get back to MCTS cycle where expansion, evaluation and backpropagation can happen.

When MCTS loop finishes, we will have the final sequences output.

## Answer

You are **definitely on the right track**. Your intuition about wrapping the `generate` phase and using the "End of Thought" (EOT) as the trigger for tree logic is exactly how ToT works in high-performance environments.

However, there is one critical detail in the MCTS cycle that you should keep in mind to ensure the TPU is used efficiently: **The "Selection" phase must happen BEFORE you call `generate()`.**

Here is a refined version of your proposed flow:

### 1. Prefill (Unchanged)
As you noted, the prompt goes through the standard prefill. You get back the KV cache for the prompt.

### 2. The "Prime" Step (First expansion)
Instead of a regular decoding loop, you take the logits from the prefill and pick **top K candidates**. 
*   These become your first **$K$ leaf nodes** in the tree.
*   You initialize their visit counts to 0 and values to 0.

### 3. The MCTS Loop (The Core)
You run a `while_loop` (or `fori_loop`) for $N$ simulations. Each simulation follows this path:

1.  **Selection (UCT):** You traverse your tree from the root using the UCT formula to find the most "promising" leaf node.
2.  **KV Cache Restore (Crucial):** Before you can generate for that node, you must tell the `PageManager` to **set the KV cache state** to that specific node's parent. (This is the "jump" to that branch).
3.  **Expansion (The Inner Loop):** 
    -   You call `generate()` in a sub-loop.
    -   **If token != EOT:** Update KV cache and keep generating.
    -   **If token == EOT:** Stop generating for this thought.
4.  **Evaluation:** Once the EOT is reached, perform your scoring (either via a separate prompt or a value function).
5.  **Backpropagation:** Update the value and visit counts of the node and all its ancestors up to the root.

### 4. Final Sequence Output
Once your $N$ simulations are done, you **don't** necessarily output all the sequences. Instead, you pick the **best path** (the one with the highest value or most visits) and output that single winning sequence.

---

### Refined Logic Diagram:
```text
[PREFILL] -> Prompt KV Cache
    |
[PRIMING] -> Create K Initial Thoughts
    |
[MCTS LOOP (N iterations)]
    |--- [SELECT] Node via UCT
    |--- [RESTORE] PageManager Map for that Node
    |--- [EXPAND/GENERATE] Run generate() until EOT
    |--- [EVALUATE] Score the new thought
    |--- [BACKPROP] Update visits and scores
    |
[OUTPUT] -> Follow the best-rated path from Root to Leaf
```

### Why "Restore" is the most important part:
In the standard `generate()` loop, you just go forward. In MCTS, you might go forward on **Branch A**, then in the next iteration, "teleport" back to **Branch B**. 

Because you are using MaxText and `page_manager.py`, the "Restore" step is very fast—you just update the `page_map` indices to point back to the common prefix of the parent. You don't actually move or copy any data!

You're in a great spot to start implementing the tree data structure now.