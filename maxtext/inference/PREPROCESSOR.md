Viewed utils.py:1-792

A **Preprocessor** is the "translator" that takes raw data from the real world—like an image file (`.jpg`) or a sound recording (`.wav`)—and turns it into a format the model can actually understand (mathematical tensors).

### 1. What does the Preprocessor do?
Large Language Models represent everything as numbers. If you are using a **Multimodal** model (one that can "see" or "hear"), the preprocessor handles the messy work of:
*   **For Images**: It opens the file, resizes it to the exact resolution the model expects (e.g., 224x224), and "normalizes" the colors (converting 0-255 RGB values into small decimals like -1.0 to 1.0).
*   **For Audio**: It converts a sound wave into a **Mel Spectrogram** (which is essentially a "picture" of the sound's frequencies over time).
*   **For Tiling**: Some advanced models (like Llama-4 or Gemma-3) break large images into smaller "tiles." The preprocessor handles this slicing.

### 2. What is `processor_outputs`?
In [decode.py](cci:7://file:///Users/kevinwang/Projects/ml/maxtext/decode.py:0:0-0:0), the line `processor_outputs = mm_utils.PreprocessorOutput()` creates a container (a dataclass) to hold the results of this processing. 

Think of it as a **folder** that contains:
*   **`pixel_values`**: The actual numbers representing the image.
*   **`pixel_mask`**: A map that tells the model, "This part is the real image, and this part is just empty padding."
*   **`audio_values`**: The numbers representing the sound.
*   **`num_images`**: A simple count of how many images were found in the prompt.

### How it's used in [decode.py](cci:7://file:///Users/kevinwang/Projects/ml/maxtext/inference/decode.py:0:0-0:0):
```python
if config.use_multimodal:
    # 1. The Preprocessor does the heavy lifting
    processor_outputs = mm_processor.preprocess_mm_data(config)
    
    # 2. Later, we hand those outputs to the Engine
    prefill_result, first_token = engine.prefill(
        ...,
        images=processor_outputs.pixel_values,
        audio_values=processor_outputs.audio_values,
        ...
    )
```

**In summary:** The preprocessor is the **bridge** between your files on disk and the model's brain. [PreprocessorOutput](cci:2://file:///Users/kevinwang/Projects/ml/maxtext/multimodal/utils.py:33:0-59:38) is the **standard package** that carries that processed data to the engine.
