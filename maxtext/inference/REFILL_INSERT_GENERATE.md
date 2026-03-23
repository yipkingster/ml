Viewed maxengine.py:800-1599

These are the three main "verbs" used by the **MaxEngine** to manage how the model thinks and speaks.

### 1. [prefill](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:591:2-650:5) (The "Read and Remember" phase)
This is always the first step.
*   **What it does**: It takes your entire prompt (text, images, and video) and processes it all at once in one "big gulp."
*   **The Output**: It creates the **KV Cache** (the model's short-term memory). 
*   **Visual**: Think of this as the model silently reading your 5-page essay before it answers. It focuses on understanding the context.

### 2. [insert](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1341:2-1367:31) (The "Copy and Paste" phase)
This is an optimization for speed.
*   **What it does**: It takes a pre-computed "Memory" (a KV cache) that you already saved earlier and "plugs" it into a slot in the current session.
*   **Why use it?**: If you have a massive "System Prompt" (like 50 pages of instructions) that stays the same for every user, you don't want to re-read it every time. You [prefill](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:591:2-650:5) it once, and then just [insert](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1341:2-1367:31) that pre-made memory into every new conversation.

### 3. [generate](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:944:2-980:64) (The "Think and Talk" phase)
This is the autoregressive loop. 
*   **What it does**: It looks at the "Memory" (KV Cache) created by the [prefill](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:591:2-650:5) or [insert](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1341:2-1367:31) steps and predicts the **very next token**.
*   **The Loop**: It takes that new token, adds it to the memory, and then runs again to predict the *next* word.
*   **Visual**: This is the model actually typing out the response on your screen, one word at a time.

---

### In summary:
1.  **[prefill](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:591:2-650:5)**: Processes a **new** prompt to create memory.
2.  **[insert](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1341:2-1367:31)**: Recycles **old** memory to save time.
3.  **[generate](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:944:2-980:64)**: Uses that memory to **create** an answer.
