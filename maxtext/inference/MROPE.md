Searched for "mrope_position_deltas = "
Viewed input_pipeline_utils.py:1-800

Actually, to keep things consistent, MaxText uses a **3D coordinate system for everything** (Text, Images, and Videos). 

Instead of switching between 2D and 3D, it treats every token as having a position in **Temporal ($T$)**, **Height ($H$)**, and **Width ($W$)** space.

### 1. How the 3D grid works for different inputs:

| Input Type | $T$ (Time) | $H$ (Height) | $W$ (Width) |
| :--- | :--- | :--- | :--- |
| **Text** | Increments $(0, 1, 2...)$ | Fixed at $0$ | Fixed at $0$ |
| **Image** | Fixed at $0$ | Grid index $(0, 1...)$ | Grid index $(0, 1...)$ |
| **Video** | Increments for each frame | Grid index $(0, 1...)$ | Grid index $(0, 1...)$ |

### 2. The Mental Model:
*   **An Image is just a "Video with 1 frame."** Its $T$ coordinate never changes.
*   **Text is just a "1-pixel Video."** Its $H$ and $W$ coordinates never change.

### 3. What the `mrope_position_deltas` array looks like:
Because of this "Unified 3D" approach, the `mrope_position_deltas` array is almost always a 2D matrix of shape [(3, N)](cci:2://file:///Users/kevinwang/Projects/ml/maxtext/configs/types.py:1625:0-1632:119), where:
*   **Row 0**: The "jumps" in the **Time** dimension.
*   **Row 1**: The "jumps" in the **Height** dimension.
*   **Row 2**: The "jumps" in the **Width** dimension.
*   **Columns ($N$)**: One column for every token in your sequence.

**In summary:** It is **always 3-Dimensional**. By using 3D for everything, the model can use one single mathematical formula (the MROPE rotation) to process text, photos, and videos all at the same time without needing special "if/else" logic for each one.