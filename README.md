Code for the paper "Localizing Reasoning Training-Induced Changes in Large Language Models" at the Mechanistic Interpretability Workshop at NeurIPS 2025.

## Setup
Install uv: https://docs.astral.sh/uv/getting-started/installation/.

Simply run python scripts with `uv run script.py`, it will install the necessary environment.


## Reproducing results


### CKA
1. Create cache for generated answers. Change `model` to any of the filenames of `conf/model` for other models.
```
uv run scripts/cache_representations.py device="cuda:0" layers_to_extract=[0] model=qwen-r1-distill

// or run these scripts to do it for all models
./scripts/start_caching.sh
./scripts/start_caching_2.sh
```

2. Compute CKA comparisons.
   Runs comparison between models descending from Qwen-Math-7B by default.
   Edit `conf/cka_same_input.yaml` under `comparison_model_selection` to compare other models.
```
uv run scripts/run_cka_analysis_with_fixed_trace.py device=cuda:0 model_devices=[cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6]
```

1. Run CKA ablations.
   Make sure to adjust the devices in the script.
   This will create multiple csv files in `outputs/cka_in_detail`, which will be used to generate the final figures.
```
./run_cka_ablations.sh
```

### Weight Comparison
Run all cells in  `weight_diff.ipynb` (or convert to a script before).
This will likely take a few hours.

### Figures
Use `paper_figs.ipynb`.
