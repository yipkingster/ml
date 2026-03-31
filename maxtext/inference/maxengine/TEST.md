# Test Structure

Based on my recursive search of the repository, there are **no traditional unit tests** (e.g., `maxengine_test.py` or `pytest` files) for the [maxengine.py](cci:7://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:0:0-0:0) or [page_manager.py](cci:7://file:///Users/kevinwang/Projects/ml/maxtext/inference/page_manager.py:0:0-0:0) source code you have been modifying.

The repository structure for testing seems to follow a different pattern:

### 1. Resource Calculation Tests
The only dedicated unit test found in the entire `inference/` directory is:
*   [test_sharding_utils.py](file:///Users/kevinwang/Projects/ml/maxtext/inference/scripts/test_sharding_utils.py)
    *   This tests the math for FLOPs, communication costs, and memory usage for various sharding strategies.

### 2. "Smoke Test" Configurations
Instead of unit tests, the repository uses specialized YAML configurations for "smoke tests." These are used to run the engine for a very small number of steps (e.g., 10 steps) with synthetic data to ensure the system doesn't crash after a change.
*   [tpu_smoke_test.yml](file:///Users/kevinwang/Projects/ml/maxtext/configs/tpu/tpu_smoke_test.yml)
*   [gpu_smoke_test.yml](file:///Users/kevinwang/Projects/ml/maxtext/configs/gpu/gpu_smoke_test.yml)

### 3. Inference Microbenchmarks
There is a microbenchmark script that can be used to verify the performance and basic functionality of the inference engine:
*   [inference_microbenchmark.py](file:///Users/kevinwang/Projects/ml/maxtext/inference/inference_microbenchmark.py)

**Summary:** 
If you want to verify your changes to [maxengine.py](cci:7://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:0:0-0:0), the standard workflow in this repo is to run a **smoke test** or the **microbenchmark** on a TPU/GPU rather than running a traditional unit test.

# Algorithm Parameter

Viewed types.py:1-800
Ran command: `grep -n "decode_sampling_strategy" /Users/kevinwang/Projects/ml/maxtext/configs/types.py`
Ran command: `grep -E "decode_num_beams|decode_num_beam_groups|decode_diversity_penalty" /Users/kevinwang/Projects/ml/maxtext/configs/types.py`

To switch the decoding method to your new **Diverse Beam Search (DBS)** algorithm, you have three primary options depending on how you are running the code:

### 1. In a Python Script (Programmatic)
If you are interacting with the [MaxEngine](cci:2://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:99:0-1842:29) object directly, the easiest way is to pass the `algorithm` argument to the [generate](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:944:2-980:64) or [_generate_jit](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1000:2-1147:13) methods:

```python
engine.generate(
    params=params,
    decode_state=state,
    algorithm="diverse_beam_search"  # This will override the config file's strategy
)
```

### 2. Via Command Line Arguments
If you are running an inference binary like [decode.py](cci:7://file:///Users/kevinwang/Projects/ml/maxtext/decode.py:0:0-0:0), you can override the configuration from the terminal:

```bash
python3 decode.py configs/base.yml \
    decode_sampling_strategy=diverse_beam_search \
    decode_num_beams=4 \
    decode_num_beam_groups=2 \
    decode_diversity_penalty=0.1
```

### 3. In the Config YAML File
If you have a dedicated test configuration file (like `configs/base.yml`), you can modify it permanently:

```yaml
# configs/my_test.yml
...
decode_sampling_strategy: "diverse_beam_search"
decode_num_beams: 4
decode_num_beam_groups: 2
decode_diversity_penalty: 0.1
...
```

### Important Parameters for DBS:
Since you integrated these into [_generate_jit](cci:1://file:///Users/kevinwang/Projects/ml/maxtext/inference/maxengine/maxengine.py:1000:2-1147:13) (lines 1077-1079), make sure these values are set correctly in your config to avoid errors:
*   **`decode_num_beams`**: Total number of candidates to track (e.g., `4`).
*   **`decode_num_beam_groups`**: Number of groups to split the beams into (e.g., `2`). Note that `decode_num_beams` must be divisible by `decode_num_beam_groups`.
*   **`decode_diversity_penalty`**: The penalty applied to ensure different groups generate diverse results (e.g., `0.1`–`1.0`).
