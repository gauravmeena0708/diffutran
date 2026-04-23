# Batch Experiment Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `scripts/batch_experiment.py` to skip re-training when checkpoints exist, support `--models` / `--datasets` / `--overwrite` CLI flags, and use correct step counts for all four sampling variants.

**Architecture:** All changes are surgical edits to a single script. A small `_needs_training()` helper is extracted for testability. CLI flags are added to the existing argparse block and threaded into `main()`. No new files are introduced except the test file.

**Tech Stack:** Python stdlib (`os`, `argparse`), pytest

---

### Task 1: Fix DIFFUTRON_CONFIGS step counts

**Files:**
- Modify: `scripts/batch_experiment.py` (lines 21-24)
- Create: `tests/test_batch_experiment.py`

The current configs use 20 steps for both DDIM and DDPM. DDIM's sweet spot is 50 steps; DDPM's noise schedule requires exactly 1000 steps — fewer steps with `eta=1.0` degrades severely because `sigma_t` is far too large for the stride size.

- [ ] **Step 1: Write failing test**

Create `tests/test_batch_experiment.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.batch_experiment import DIFFUTRON_CONFIGS, DSFLOW_CONFIGS


def test_diffutron_ddim_steps():
    ddim = next(c for c in DIFFUTRON_CONFIGS if c["label"] == "diffutron-ddim")
    assert ddim["steps"] == 50
    assert ddim["eta"] == 0.0


def test_diffutron_ddpm_steps():
    ddpm = next(c for c in DIFFUTRON_CONFIGS if c["label"] == "diffutron-ddpm")
    assert ddpm["steps"] == 1000
    assert ddpm["eta"] == 1.0


def test_dsflow_dpmpp_steps():
    dpm = next(c for c in DSFLOW_CONFIGS if c["label"] == "dsflow-dpm++")
    assert dpm["steps"] == 10
    assert dpm["solver"] == "dpm++"


def test_dsflow_euler_steps():
    euler = next(c for c in DSFLOW_CONFIGS if c["label"] == "dsflow-euler")
    assert euler["steps"] == 20
    assert euler["solver"] == "euler"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/gaurav/syntab/base_test2/sub/diffutran
pytest tests/test_batch_experiment.py::test_diffutron_ddim_steps tests/test_batch_experiment.py::test_diffutron_ddpm_steps -v
```

Expected: FAIL — both assert wrong step values (currently 20).

- [ ] **Step 3: Fix the configs**

In `scripts/batch_experiment.py`, replace lines 21-24:

```python
DIFFUTRON_CONFIGS = [
    {"eta": 0.0, "steps": 50,   "label": "diffutron-ddim"},
    {"eta": 1.0, "steps": 1000, "label": "diffutron-ddpm"},
]
```

- [ ] **Step 4: Run all four config tests**

```bash
pytest tests/test_batch_experiment.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/batch_experiment.py tests/test_batch_experiment.py
git commit -m "fix: correct DDIM to 50 steps and DDPM to 1000 steps"
```

---

### Task 2: Extract `_needs_training` helper

**Files:**
- Modify: `scripts/batch_experiment.py` — add helper before `run_command`
- Modify: `tests/test_batch_experiment.py` — add tests

- [ ] **Step 1: Write failing tests**

Append to `tests/test_batch_experiment.py`:

```python
import tempfile
from scripts.batch_experiment import _needs_training


def test_needs_training_missing_artifact():
    with tempfile.TemporaryDirectory() as d:
        assert _needs_training(d, ["model.pt"]) is True


def test_needs_training_all_present():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "model.pt"), "w").close()
        open(os.path.join(d, "tokenizer.pt"), "w").close()
        assert _needs_training(d, ["model.pt", "tokenizer.pt"]) is False


def test_needs_training_partial():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "model.pt"), "w").close()
        # tokenizer.pt is absent
        assert _needs_training(d, ["model.pt", "tokenizer.pt"]) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_batch_experiment.py::test_needs_training_missing_artifact -v
```

Expected: FAIL with `ImportError: cannot import name '_needs_training'`.

- [ ] **Step 3: Add the helper to `batch_experiment.py`**

Insert after the `DIFFUTRON_CONFIGS` block (before `def run_command`):

```python
def _needs_training(model_dir: str, artifacts: list) -> bool:
    """Return True if any required checkpoint artifact is missing."""
    return not all(os.path.exists(os.path.join(model_dir, a)) for a in artifacts)
```

- [ ] **Step 4: Run the new tests**

```bash
pytest tests/test_batch_experiment.py -v
```

Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/batch_experiment.py tests/test_batch_experiment.py
git commit -m "feat: extract _needs_training helper with tests"
```

---

### Task 3: Add `--models`, `--datasets`, `--overwrite` CLI flags

**Files:**
- Modify: `scripts/batch_experiment.py` — argparse block
- Modify: `tests/test_batch_experiment.py` — add CLI parse tests

- [ ] **Step 1: Write failing tests**

Append to `tests/test_batch_experiment.py`:

```python
import argparse


def _parse(args_list):
    """Helper: parse args the same way __main__ does."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--models', nargs='+', default=['diffutron', 'dsflow'],
                        choices=['diffutron', 'dsflow'])
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(args_list)


def test_default_models_both():
    args = _parse([])
    assert set(args.models) == {'diffutron', 'dsflow'}


def test_models_single():
    args = _parse(['--models', 'dsflow'])
    assert args.models == ['dsflow']


def test_datasets_default_none():
    args = _parse([])
    assert args.datasets is None


def test_datasets_explicit():
    args = _parse(['--datasets', 'adult', 'shoppers'])
    assert args.datasets == ['adult', 'shoppers']


def test_overwrite_default_false():
    args = _parse([])
    assert args.overwrite is False


def test_overwrite_flag():
    args = _parse(['--overwrite'])
    assert args.overwrite is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_batch_experiment.py::test_default_models_both -v
```

Expected: FAIL (the real parser doesn't have `--models` yet).

- [ ] **Step 3: Add the flags to the argparse block in `batch_experiment.py`**

Replace the `if __name__ == '__main__':` block:

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help="1 epoch, 100 samples, d_model=128 for fast smoke-testing.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Training epochs per model.")
    parser.add_argument('--models', nargs='+', default=['diffutron', 'dsflow'],
                        choices=['diffutron', 'dsflow'],
                        help="Model families to run (default: both).")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help="Datasets to run (default: all in workdir/data/processed/).")
    parser.add_argument('--overwrite', action='store_true',
                        help="Re-train even if checkpoint exists (default: skip if trained).")
    args = parser.parse_args()
    main(args)
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_batch_experiment.py -v
```

Expected: 13 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/batch_experiment.py tests/test_batch_experiment.py
git commit -m "feat: add --models, --datasets, --overwrite CLI flags"
```

---

### Task 4: Wire `--datasets` filter and training skip logic

**Files:**
- Modify: `scripts/batch_experiment.py` — `main()` function

This task wires three behaviours into `main()`:
1. Filter `datasets` list by `args.datasets` before the loop.
2. Skip DS-Flow training if checkpoint exists (unless `--overwrite`).
3. Skip DiffuTron training if checkpoint exists (unless `--overwrite`).

- [ ] **Step 1: Add `--datasets` filter after the datasets list is built**

In `main()`, replace:

```python
datasets = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
```

with:

```python
datasets = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
if args.datasets:
    missing = set(args.datasets) - set(datasets)
    if missing:
        print(f"Warning: requested datasets not found: {sorted(missing)}")
    datasets = [d for d in datasets if d in args.datasets]
```

- [ ] **Step 2: Wire skip logic for DS-Flow training**

In `main()`, replace the DS-Flow training block:

```python
        # 1. Train DS-Flow
        ds_model_dir = os.path.join(base_model_dir, ds, "dsflow")
        train_cmd = [
            "python", "-m", "src.flow_model.train",
            "--data_path", data_path,
            "--save_dir", ds_model_dir,
            "--num_epochs", "1" if args.debug else str(args.epochs),
            "--batch_size", "256",
            "--d_model", "128" if args.debug else "512"
        ]
        if cat_cols:
            train_cmd.extend(["--categorical_cols"] + cat_cols)
        if target_col:
            train_cmd.extend(["--condition_col", target_col])
        if is_temporal:
            train_cmd.extend(["--is_temporal", "--sequence_len", str(seq_len)])
            
        run_command(train_cmd, f"Training DS-Flow on {ds}")
```

with:

```python
        # 1. Train DS-Flow
        ds_model_dir = os.path.join(base_model_dir, ds, "dsflow")
        if args.overwrite or _needs_training(ds_model_dir, ["model_final.pt", "tokenizer.pt"]):
            train_cmd = [
                "python", "-m", "src.flow_model.train",
                "--data_path", data_path,
                "--save_dir", ds_model_dir,
                "--num_epochs", "1" if args.debug else str(args.epochs),
                "--batch_size", "256",
                "--d_model", "128" if args.debug else "512"
            ]
            if cat_cols:
                train_cmd.extend(["--categorical_cols"] + cat_cols)
            if target_col:
                train_cmd.extend(["--condition_col", target_col])
            if is_temporal:
                train_cmd.extend(["--is_temporal", "--sequence_len", str(seq_len)])
            run_command(train_cmd, f"Training DS-Flow on {ds}")
        else:
            print(f"[skip] DS-Flow checkpoint found for {ds} — use --overwrite to retrain")
```

- [ ] **Step 3: Wire skip logic for DiffuTron training**

Replace the DiffuTron training block:

```python
        # 2. Train DiffuTron
        m1_model_dir = os.path.join(base_model_dir, ds, "diffutron")
        if not is_temporal:
            train_m1_cmd = [
                "python", "scripts/train.py",
                "--data_path", data_path,
                "--save_dir", m1_model_dir,
                "--num_epochs", "1" if args.debug else str(args.epochs),
                "--batch_size", "256",
                "--hidden_dim", "128" if args.debug else "256"
            ]
            if cat_cols:
                train_m1_cmd.extend(["--categorical_cols"] + cat_cols)
            if target_col:
                train_m1_cmd.extend(["--condition_col", target_col])
                
            run_command(train_m1_cmd, f"Training DiffuTron on {ds}")
```

with:

```python
        # 2. Train DiffuTron
        m1_model_dir = os.path.join(base_model_dir, ds, "diffutron")
        if not is_temporal:
            if args.overwrite or _needs_training(m1_model_dir, ["checkpoint_epoch_final.pt"]):
                train_m1_cmd = [
                    "python", "scripts/train.py",
                    "--data_path", data_path,
                    "--save_dir", m1_model_dir,
                    "--num_epochs", "1" if args.debug else str(args.epochs),
                    "--batch_size", "256",
                    "--hidden_dim", "128" if args.debug else "256"
                ]
                if cat_cols:
                    train_m1_cmd.extend(["--categorical_cols"] + cat_cols)
                if target_col:
                    train_m1_cmd.extend(["--condition_col", target_col])
                run_command(train_m1_cmd, f"Training DiffuTron on {ds}")
            else:
                print(f"[skip] DiffuTron checkpoint found for {ds} — use --overwrite to retrain")
```

- [ ] **Step 4: Run the test suite**

```bash
pytest tests/test_batch_experiment.py -v
```

Expected: 13 PASS (no regressions; these changes are not covered by unit tests — they require filesystem side-effects — but the helper and CLI tests still validate the building blocks).

- [ ] **Step 5: Commit**

```bash
git add scripts/batch_experiment.py
git commit -m "feat: skip training when checkpoints exist, add --overwrite flag"
```

---

### Task 5: Wire `--models` filter into training and sampling loops

**Files:**
- Modify: `scripts/batch_experiment.py` — `main()` function

- [ ] **Step 1: Wrap DS-Flow training in `--models` guard**

The DS-Flow training block (Task 4, Step 2 result) is currently unconditional. Wrap it:

```python
        # 1. Train DS-Flow
        ds_model_dir = os.path.join(base_model_dir, ds, "dsflow")
        if 'dsflow' in args.models:
            if args.overwrite or _needs_training(ds_model_dir, ["model_final.pt", "tokenizer.pt"]):
                train_cmd = [
                    "python", "-m", "src.flow_model.train",
                    "--data_path", data_path,
                    "--save_dir", ds_model_dir,
                    "--num_epochs", "1" if args.debug else str(args.epochs),
                    "--batch_size", "256",
                    "--d_model", "128" if args.debug else "512"
                ]
                if cat_cols:
                    train_cmd.extend(["--categorical_cols"] + cat_cols)
                if target_col:
                    train_cmd.extend(["--condition_col", target_col])
                if is_temporal:
                    train_cmd.extend(["--is_temporal", "--sequence_len", str(seq_len)])
                run_command(train_cmd, f"Training DS-Flow on {ds}")
            else:
                print(f"[skip] DS-Flow checkpoint found for {ds} — use --overwrite to retrain")
```

- [ ] **Step 2: Wrap DiffuTron training in `--models` guard**

The DiffuTron training block (Task 4, Step 3 result). Change the outer condition from `if not is_temporal:` to combine both guards:

```python
        # 2. Train DiffuTron
        m1_model_dir = os.path.join(base_model_dir, ds, "diffutron")
        if 'diffutron' in args.models and not is_temporal:
            if args.overwrite or _needs_training(m1_model_dir, ["checkpoint_epoch_final.pt"]):
                train_m1_cmd = [
                    "python", "scripts/train.py",
                    "--data_path", data_path,
                    "--save_dir", m1_model_dir,
                    "--num_epochs", "1" if args.debug else str(args.epochs),
                    "--batch_size", "256",
                    "--hidden_dim", "128" if args.debug else "256"
                ]
                if cat_cols:
                    train_m1_cmd.extend(["--categorical_cols"] + cat_cols)
                if target_col:
                    train_m1_cmd.extend(["--condition_col", target_col])
                run_command(train_m1_cmd, f"Training DiffuTron on {ds}")
            else:
                print(f"[skip] DiffuTron checkpoint found for {ds} — use --overwrite to retrain")
```

- [ ] **Step 3: Wrap DS-Flow sampling loop in `--models` guard**

The DS-Flow sampling loop currently starts with `for cfg in DSFLOW_CONFIGS:`. Wrap it:

```python
        if 'dsflow' in args.models:
            for cfg in DSFLOW_CONFIGS:
                variant_dir = os.path.join(syn_ds_dir, cfg["label"])
                os.makedirs(variant_dir, exist_ok=True)
                start = time.time()
                cmd = [
                    "python", "-m", "src.flow_model.sampler",
                    "--checkpoint_path", os.path.join(ds_model_dir, "model_final.pt"),
                    "--tokenizer_path", os.path.join(ds_model_dir, "tokenizer.pt"),
                    "--dataset_name", ds,
                    "--output_dir", variant_dir,
                    "--num_samples", str(n_samp),
                    "--steps", str(cfg["steps"]),
                    "--solver", cfg["solver"],
                    "--d_model", "128" if args.debug else "512",
                ]
                if target_col:
                    cmd.extend(["--condition_col", target_col])
                if constraint_col and constraint_val:
                    cmd.extend(["--constraint", f"{constraint_col}={constraint_val}"])
                run_command(cmd, f"Sampling {cfg['label']} for {ds}")
                eff = (time.time() - start) / (n_samp / 1000)
                files = sorted(f for f in os.listdir(variant_dir) if f.endswith(".csv"))
                if files:
                    variant_outputs.append((cfg["label"], os.path.join(variant_dir, files[-1]), eff))
```

- [ ] **Step 4: Wrap DiffuTron sampling loop in `--models` guard**

The DiffuTron sampling loop currently starts with `if not is_temporal:`. Combine guards:

```python
        if 'diffutron' in args.models and not is_temporal:
            for cfg in DIFFUTRON_CONFIGS:
                variant_dir = os.path.join(syn_ds_dir, cfg["label"])
                os.makedirs(variant_dir, exist_ok=True)
                start = time.time()
                cmd = [
                    "python", "scripts/generate_test.py",
                    "--checkpoint_path", os.path.join(m1_model_dir, "checkpoint_epoch_final.pt"),
                    "--preprocessor_path", os.path.join(m1_model_dir, "preprocessor.json"),
                    "--condition_encoder_path", os.path.join(m1_model_dir, "condition_encoder.json"),
                    "--real_data_path", data_path,
                    "--output_dir", variant_dir,
                    "--num_samples", str(n_samp),
                    "--num_inference_steps", str(cfg["steps"]),
                    "--eta", str(cfg["eta"]),
                    "--hidden_dim", "128" if args.debug else "256",
                ]
                if target_col:
                    cmd.extend(["--condition_col", target_col])
                if constraint_col and constraint_val:
                    cmd.extend(["--constraint", f"{constraint_col}={constraint_val}"])
                run_command(cmd, f"Sampling {cfg['label']} for {ds}")
                eff = (time.time() - start) / (n_samp / 1000)
                files = sorted(f for f in os.listdir(variant_dir) if f.endswith(".csv"))
                if files:
                    variant_outputs.append((cfg["label"], os.path.join(variant_dir, files[-1]), eff))
```

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/test_batch_experiment.py -v
```

Expected: 13 PASS.

- [ ] **Step 6: Smoke test the CLI flags parse cleanly**

```bash
python scripts/batch_experiment.py --help
```

Expected: help text shows `--models`, `--datasets`, `--overwrite`, `--epochs`, `--debug` with correct defaults.

- [ ] **Step 7: Commit**

```bash
git add scripts/batch_experiment.py
git commit -m "feat: --models filter gates training and sampling per model family"
```

---

## Self-Review

**Spec coverage:**
- ✅ Skip re-training by default — Task 4 wires `_needs_training()` checks
- ✅ `--overwrite` flag — Task 3 adds it; Task 4 uses it
- ✅ `--models` flag — Task 3 adds it; Task 5 wires it into both training and sampling
- ✅ `--datasets` flag — Task 3 adds it; Task 4 filters the dataset list
- ✅ DDIM 50 steps — Task 1
- ✅ DDPM 1000 steps — Task 1
- ✅ DPM++ 10 steps, Euler 20 steps — Task 1 (already correct, test verifies)

**Placeholder scan:** No TBDs. All code blocks are complete. All commands include expected output.

**Type consistency:** `_needs_training(model_dir: str, artifacts: list) -> bool` used consistently across Task 2 definition and Task 4 call sites. `args.models` is always a `list[str]`, checked with `'dsflow' in args.models` consistently.
