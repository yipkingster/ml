Viewed processor_gemma3.py:1-274

Actually, [image_offsets](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/multimodal/processor.py:65:0-80:12) accounts for **all images** you provide, not just one.

If you provide 3 images, [image_offsets](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/multimodal/processor.py:65:0-80:12) will be the sum of the tokens required for all 3. Here is how it works under the hood (using **Gemma3** as an example):

### 1. Multi-Image Calculation
Inside the preprocessor, it checks how many images were loaded:
```python
# From processor_gemma3.py
num_images = processor_output.pixel_values.shape[0] 
return (tokens_per_image - 1) * num_images
```
So if 1 image = 260 tokens, then:
*   1 Image $\rightarrow$ [image_offsets](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/multimodal/processor.py:65:0-80:12) = **260**
*   2 Images $\rightarrow$ [image_offsets](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/multimodal/processor.py:65:0-80:12) = **520**
*   3 Images $\rightarrow$ [image_offsets](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/multimodal/processor.py:65:0-80:12) = **780**

### 2. Does `prefill_length -= image_offsets` work for multiple images?
**Yes.** Because [image_offsets](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/multimodal/processor.py:65:0-80:12) is the *total* for all images, subtracting it from [prefill_length](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1695:2-1698:54) correctly reserves a "block" of space for every single image you included.

### 3. How do multiple images fit in the prompt?
When you call `mm_processor.reformat_prompt`, it looks at your text.
*   If you manually put three placeholders (like `[image][image][image]`), it replaces each one.
*   If you only have one placeholder but provided 3 images, it will actually **auto-insert** the missing placeholders at the beginning of your prompt to make sure all 3 images have a "seat" in the sequence.

**In summary:** [image_offsets](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/multimodal/processor.py:65:0-80:12) is a **cumulative total**. Whether you have one image or ten, that single line of code correctly calculates the total space they will occupy in the model's memory.
