# Computing Degeneration Axes

This pipeline mirrors the Assistant Axis flow, but builds **degeneration axes** for multiple failure modes:
- repetition loops
- verbosity
- refusal spirals
- hallucination
- contradiction

Each category produces paired responses:
- `good`: intended to be high-quality
- `degen`: biased toward the failure mode

The axis for each category is computed as:
```
axis = mean(good) - mean(degen)
```
Higher projections indicate **less degeneration** for that category.

## Requirements
- GPU with sufficient memory for your target model
- `OPENAI_API_KEY` (for the judge)

## Steps

### 1) Generate paired responses
```
uv run pipeline/degeneration/1_generate.py \
  --model google/gemma-2-27b-it \
  --output_dir outputs/gemma-2-27b/degeneration/responses
```

### 2) Extract activations
Reuse the existing activation script:
```
uv run pipeline/2_activations.py \
  --model google/gemma-2-27b-it \
  --responses_dir outputs/gemma-2-27b/degeneration/responses \
  --output_dir outputs/gemma-2-27b/degeneration/activations \
  --batch_size 16
```

### 3) Score responses (judge)
Reuse the existing judge script with degeneration categories:
```
uv run pipeline/3_judge.py \
  --responses_dir outputs/gemma-2-27b/degeneration/responses \
  --roles_dir data/degeneration/categories \
  --output_dir outputs/gemma-2-27b/degeneration/scores \
  --judge_model gpt-4.1-mini
```

Score scale for degeneration (per category):
- 0 = severe degeneration
- 1 = noticeable degeneration
- 2 = minor issues
- 3 = clean / high-quality

### 4) Compute per-category vectors
```
uv run pipeline/degeneration/4_vectors.py \
  --activations_dir outputs/gemma-2-27b/degeneration/activations \
  --scores_dir outputs/gemma-2-27b/degeneration/scores \
  --output_dir outputs/gemma-2-27b/degeneration/vectors
```

### 5) Compute axes
```
uv run pipeline/degeneration/5_axes.py \
  --vectors_dir outputs/gemma-2-27b/degeneration/vectors \
  --output outputs/gemma-2-27b/degeneration/axes.pt
```

The output file contains:
- `axes`: per-category axes
- `aggregate`: mean axis across categories

## Steering / Capping
Use `ActivationSteering` with one or more axes. For example, to cap repetition and verbosity at a chosen layer:

```python
from assistant_axis import ActivationSteering
axes = torch.load("outputs/.../axes.pt")
repetition_axis = axes["axes"]["repetition"]
verbosity_axis = axes["axes"]["verbosity"]

with ActivationSteering(
    model,
    steering_vectors=[repetition_axis[layer], verbosity_axis[layer]],
    coefficients=[0.5, 0.5],
    layer_indices=[layer, layer],
    intervention_type="addition",
):
    ...
```

For capping, re-use the `intervention_type="capping"` path and supply thresholds.
