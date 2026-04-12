In MaxText, `images` and `image_masks` are the primary inputs for **multimodal** models (like Gemma 3 or Gemini-style architectures). They allow the model to "see" visual data alongside text.

Here is exactly what they are and how they operate:

### **1. `images` (Visual Data)**
*   **What it is:** A 5-dimensional JAX array of pixel values.
*   **Shape:** `[Batch, Num_Images, Channels, Height, Width]`
*   **How it works:**
    *   **Normalization:** Raw pixels are usually normalized to a range (like -1 to 1 or 0 to 1).
    *   **Vision Encoding:** Before the visual data reaches the LLM's brain, it is passed through a **Vision Encoder** (typically a Vision Transformer or ViT). 
    *   **Tokenization:** the encoder turns each image into a sequence of "image tokens" (embeddings). These tokens are then conceptually "injected" into the prompt wherever the `[IMG]` placeholder tokens appear.

### **2. `image_masks` (Validity Metadata)**
*   **What it is:** A boolean or integer mask that identifies valid visual data within a batch.
*   **Shape:** It usually matches the `[Batch, Num_Images]` or `[Batch, Num_Images, Num_Tiles]`.
*   **How it works:**
    *   **Padding Handling:** If you are processing two requests in a batch and User A has 3 images while User B only has 1, the `images` tensor must be padded to accommodate 3 images for both. The `image_mask` will be `[True, True, True]` for User A and `[True, False, False]` for User B, telling the model to ignore the padding slots.
    *   **Resolution Tiling:** Modern models often split high-resolution images into several smaller **tiles** to preserve detail. If an image is split into a 2x2 grid (4 tiles) but isn't perfectly square, the `image_mask` denotes which tiles are real image content and which are just filler.

### **How They Work Together in the Pipeline**
1.  **Detection:** In [maxengine.py](cci:7://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/maxengine/maxengine.py:0:0-0:0), the code checks if `images` is provided. 
2.  **Preprocessing:** A dedicated `multimodal_processor` (like `Gemma3Processor`) handles resizing and tiling the images based on the model's requirements.
3.  **Forward Pass**:
    *   The `VisionEncoder` processes only the entries marked `True` in `image_masks`.
    *   These visual embeddings are interleaved with the text embeddings.
    *   The LLM's **Attention Mechanism** uses the `image_masks` (via the `sequence_indicator`) to ensure it doesn't "attend" to garbage pixels or padding tiles.

### **Summary**
*   **`images`** are the **pixels** (the "what").
*   **`image_masks`** are the **instructions** for the model about which pixels are real and which should be ignored (the "where").
