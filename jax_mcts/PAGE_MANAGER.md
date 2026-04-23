# Page Manager
The [page_manager.py](cci:7://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:0:0-0:0) file is the heart of MaxText’s **Paged Attention** system. Think of it as a **Virtual Memory Manager** for the LLM's KV cache. It allows MaxText to handle multiple requests of varying lengths without wasting memory on huge, pre-allocated static buffers.

Here is a breakdown of the code and how it works:

### 1. The Core Data Structure: [PageState](cci:2://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:45:0-61:35)
At its center is the [PageState](cci:2://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:45:0-61:35) dataclass. It doesn't store the actual KV cache data (the tokens' keys and values), but rather the **metadata** that tells the system where that data is stored in the global physical memory pool.

*   `page_status` (`[num_pages]`): A global array where `0` means the page is free and `1` means it's taken.
*   `page_map` (`[max_groups, pages_per_group]`): This is the **Page Table**. For each request (group), it stores the list of physical indices in the global pool that belong to it.
*   `active_page`: The index of the specific page currently being written to for a particular request.
*   `active_page_position`: Where within that page the next token should go (until the page is full).

---

### 2. How Paging Works (The Life Cycle)

Broadly, the code follows a three-step cycle for every request:

#### A. Reservation (Prefill Phase)
When a new prompt arrives, [_reserve_pages_for_group](cci:1://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:179:0-283:20) is called.
1.  It calculates how many pages the prompt needs based on `true_length`.
2.  It finds free slots in `page_status`.
3.  It updates the `page_map` for that request so the Attention layers know where to look.

#### B. Dynamic Growth (Decode Phase)
As the model generates tokens one-by-one, [_update_decode_pages_global](cci:1://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:303:0-385:20) handles memory growth:
1.  Every time a sequence grows, it checks if it has crossed a page boundary (e.g., if a page holds 16 tokens and we just generated the 17th).
2.  If a new page is needed, it dynamically allocates one from the pool and adds it to that request's `page_map`.

#### C. Release
Once a request is finished (e.g., it hits `<EOS>`), [_release_pages_for_group](cci:1://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:130:0-176:3) marks those physical pages as `0` (free) in `page_status` so they can be reused immediately by other requests.

---

### 3. Key JAX Optimizations
*   **Vectorization**: Most functions like [_update_decode_pages_global](cci:1://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:303:0-385:20) use `jax.lax.fori_loop` or vectorized masking. This allows the TPU to manage memory for hundreds of requests simultaneously in a single JITTED operation.
*   **Page 0 Workaround**: You'll see comments about `page 0`. By convention, page 0 is often reserved or skipped to avoid "null pointer" style errors where an uninitialized index points to real data.

---

### 4. Why this matters for your MCTS Project
In your **Tree of Thoughts** implementation, this is where the magic of **Prefix Sharing** happens:

In a normal MCTS, when you "branch" from a parent node to multiple children, you don't want to copy the parent's KV cache. With this [PageManager](cci:2://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:388:0-594:5), you can simply:
1.  Find the `page_map` for the parent node.
2.  Copy that map to the child nodes.
3.  Because the [PageManager](cci:2://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:388:0-594:5) just tracks indices, multiple children can "point" to the same physical pages for their shared past, only allocating *new* pages for their unique future thoughts.

**In summary:** [page_manager.py](cci:7://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:0:0-0:0) is the bookkeeper that ensures memory is used efficiently, allowing you to run complex, branching searches like MCTS without running out of TPU memory.

# Page Group

In MaxText, a **Page Group** is effectively a **logical slot** in the batch.

To answer your question directly: **Yes, typically 1 Page Group = 1 Request/Prompt.**

Here is the deeper breakdown of how they are organized:

### 1. Relationship to Batching
If you configure MaxText with a `global_batch_size` of **32**, the [PageManager](cci:2://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:388:0-594:5) will create **32 Page Groups**.
*   Each index ($0$ to $31$) is a `page_group_id`.
*   These correspond exactly to the **batch dimension** of your activations and KV cache tensors.

### 2. What a Page Group actually "owns"
A Page Group is the "owner" of a set of physical memory pages. Inside the code, this is tracked in the `page_map`:
*   `page_map[group_id]` is a list of physical page indices.
*   **Prompt 1** might be assigned to **Page Group 0** and use physical pages `[5, 12, 42]`.
*   **Prompt 2** might be assigned to **Page Group 1** and use physical pages `[7, 89]`.

### 3. Why call it a "Group" instead of just a "Slot"?
The term "Group" is used because paged attention (vLLM style) allows for **Memory Sharing**:
*   In **Beam Search**, multiple "Beams" might start from the same prompt.
*   You could have multiple Page Groups that all point to the *same* physical pages for the shared prefix (the prompt), but have their own private pages for their unique generated tokens.

### 4. How it applies to your MCTS Project
This is where Page Groups become very powerful for you:
*   In your **Tree of Thoughts (ToT)**, every node in your tree could be assigned to its own `page_group_id`.
*   If you want to expand 8 different thoughts in parallel, you would assign each one to a different **Page Group** slot.
*   The [PageManager](cci:2://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/page_manager.py:388:0-594:5) ensures that even though these 8 thoughts are technically in different batch slots, they aren't wasting memory by duplicating the KV cache of the parent nodes they share.

**Summary:** 1 Page Group is essentially 1 slot in your current processing batch. Usually, it's 1 prompt, but in branching search (like MCTS), it's 1 "search path" or "thought."