# Batch Experiment Improvements — Design Spec
**Date:** 2026-04-23
**File:** `scripts/batch_experiment.py`

---

## Problem

The current `batch_experiment.py` has four pain points:

1. **Re-trains on every run** — no checkpoint detection; 9 datasets × 2 models = 18 full training runs even when nothing changed.
2. **No model/dataset selection** — must run all datasets and both model families every time.
3. **Wrong DDPM step count** — `diffutron-ddpm` used 20 steps with `eta=1.0`, which injects noise far too large for the stride size and produces garbage. Real DDPM requires 1000 steps.
4. **Only one variant per model** — the script ran a single hardcoded config per model family instead of the full set of meaningful sampling variants.

---

## Design

### Sampling Variant Configs

Four variants, hardcoded as module-level dicts (easy to hand-edit, no new file dependency):

```python
DSFLOW_CONFIGS = [
    {"solver": "dpm++", "steps": 10, "label": "dsflow-dpm++"},
    {"solver": "euler",  "steps": 20, "label": "dsflow-euler"},
]
DIFFUTRON_CONFIGS = [
    {"eta": 0.0, "steps": 50,   "label": "diffutron-ddim"},
    {"eta": 1.0, "steps": 1000, "label": "diffutron-ddpm"},
]
```

Step counts rationale:
- **diffutron-ddim / 50 steps**: DDIM paper sweet spot; 50 steps ≈ 1000-step DDPM quality.
- **diffutron-ddpm / 1000 steps**: DDPM's noise schedule is designed for exactly 1000 steps; fewer steps with `eta=1.0` degrades severely.
- **dsflow-dpm++ / 10 steps**: 2nd-order midpoint halves step requirement; flow matching paths are straighter than diffusion, so 10 is already high quality.
- **dsflow-euler / 20 steps**: 1st-order needs ~2× steps vs DPM++ to match quality.

### CLI Interface

```
python scripts/batch_experiment.py [options]

Options:
  --epochs INT          Training epochs per model (default: 10)
  --datasets DS [DS …]  Datasets to run (default: all found in workdir/data/processed/)
  --models MODEL [MODEL …]
                        Model families to run: diffutron, dsflow (default: both)
  --overwrite           Re-train even if a checkpoint already exists (default: skip)
  --debug               1 epoch, 100 samples, d_model=128 for fast smoke-testing
```

Examples:
```bash
# All datasets, all 4 variants, skip already-trained models
python scripts/batch_experiment.py --epochs 100

# Only dsflow on two datasets, force retrain
python scripts/batch_experiment.py --models dsflow --datasets adult shoppers --overwrite

# DiffuTron only, specific datasets
python scripts/batch_experiment.py --models diffutron --datasets german compas
```

### Skip / Overwrite Logic

Training is skipped by default when checkpoints exist. `--overwrite` forces a retrain.

| Model | Checkpoint path checked |
|---|---|
| DS-Flow | `workdir/models/{ds}/dsflow/model_final.pt` AND `tokenizer.pt` |
| DiffuTron | `workdir/models/{ds}/diffutron/checkpoint_epoch_final.pt` |

Sampling always runs for every variant (variant outputs live in their own subdirectories, so re-sampling is cheap and idempotent — it just overwrites the last run's file for that variant).

### Output Structure

Each variant writes to its own subdirectory to avoid filename collisions (all samplers use the same `{model}_{dataset}_{hash}_{month}_{n}.csv` pattern):

```
workdir/synthetic/{dataset}/
    dsflow-dpm++/
        flow_model_{dataset}_{hash}_{month}_{n}.csv
    dsflow-euler/
        flow_model_{dataset}_{hash}_{month}_{n}.csv
    diffutron-ddim/
        diffutron_{dataset}_{hash}_{month}_{n}.csv
    diffutron-ddpm/
        diffutron_{dataset}_{hash}_{month}_{n}.csv
```

### Results CSV

One row per `(dataset, method)` pair. `method` is the variant label (`dsflow-dpm++`, `diffutron-ddim`, etc.). Columns unchanged: `dataset, method, avg_ks, tstr, csr, privacy_mia, diversity_cov, efficiency_1k_sec`.

Temporal datasets (`"temporal" in ds.lower()`) produce only 2 rows (DS-Flow variants only) — DiffuTron is skipped, same as before.

### Filtering by `--models`

When `--models diffutron` is passed, the DS-Flow training and all `DSFLOW_CONFIGS` sampling loops are skipped entirely for every dataset. Similarly for `--models dsflow`. Both is the default.

### What Is NOT Changed

- Training commands (same flags, same model sizes)
- Evaluation command (same inline Python subprocess)
- Result aggregation and CSV save
- Debug mode behaviour
- Constraint selection (first categorical column mode)
- Temporal detection logic
