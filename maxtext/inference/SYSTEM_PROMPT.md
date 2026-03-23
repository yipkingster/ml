This is where the magic of **Efficiency** happens. 

### 1. Is the KV Cache empty?
*   **Before Prefill**: Yes. At the very start of a request, the model's memory (KV Cache) for that slot is filled with zeros.
*   **During Prefill**: No. The model starts "filling" the cache as it reads your tokens.
*   **After Prefill**: No. The cache is now **populated** with the "keys and values" (the context) of your prompt.

### 2. How the System Prompt stays the same for all users
In a large-scale system, you use a technique called **"Pre-computed Prefill."**

Imagine your system prompt is: 
`"You are a helpful AI assistant. Always be polite and never share secrets."` (50 tokens).

Instead of re-reading this for **User A**, **User B**, and **User C**:
1.  **Step 1 (Cold Start)**: The system runs a [prefill](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:591:2-650:5) on JUST those 50 tokens **once**.
2.  **Step 2 (Save)**: It saves that resulting KV Cache (let's call it `SYSTEM_KV_CACHE`).
3.  **Step 3 (User A arrives)**: 
    *   The system takes an empty memory slot for User A.
    *   It **`inserts`** the `SYSTEM_KV_CACHE` into the first 50 positions of that slot.
    *   It then only has to [prefill](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:591:2-650:5) User A’s actual question, like `"How do I boil an egg?"`.
    *   The model "feels" like it already read the system prompt because its context is already sitting in its memory!

### 3. What is *that* system prompt?
It’s simply **Text**. It is usually defined by the AI developer to set the "personality" and "rules" of the model. For the models you are looking at (like Qwen or Gemma), it often looks like this internally:

```text
<|im_start|>system
You are a helpful, harmless, and honest assistant. 
Include the current date (2025-03-22) in your response.
<|im_end|>
```

**In summary:** [prefill](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:591:2-650:5) and [insert](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1341:2-1367:31) are the ways the system **fills** the KV Cache so the model doesn't have to re-read common instructions every single time. It turns a "forgetful" model into a "prepared" one.
