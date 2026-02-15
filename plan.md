# Role
Act as a Senior Research Engineer at a top-tier AI lab (DeepMind/OpenAI level). Your goal is to implement a novel, state-of-the-art framework for Tabular Data Generation called **"DS-Flow" (Dual-Stream Flow Matching)**.

# Context
We are rejecting standard Diffusion (DDPM) and standard Tabular GANs.
We are rejecting naive "One-Hot" encodings that crash memory.
We are building a **Novel Architecture** that combines:
1.  **Dual-Stream Transformer:** Maintains both a "Global Row Vector" (holistic view) and "Local Column Tokens" (detailed view) simultaneously.
2.  **Flow Matching:** Uses Optimal Transport Conditional Flow Matching (OT-CFM) instead of Diffusion for 10x faster generation.
3.  **Quantile Tokenization:** Strictly uses Quantile Transforms for numerical data to ensure Gaussian-like inputs.

# Project Specification

## 1. Directory Structure
Create a modular, research-grade Python package:
```text
/ds_flow
    /data
        tokenizer.py       # Handling Quantile + Embeddings
        loader.py          # Data loading logic
    /models
        dual_stream.py     # The Novel Backbone (Global + Local Streams)
        flow_matching.py   # The ODE Trainer & Sampler
    /evaluation
        metrics.py         # SOTA Metrics (Alpha/Beta, CatBoost)
        comparison.py      # Benchmarking script
    config.py              # Hyperparameters
    train.py               # Main training loop
    run_benchmark.py       # Script to compare DS-Flow vs TabDiff/CuTS

```

## 2. Component Details (Strict Requirements)

### A. Data Layer (`tokenizer.py`)

* **Input:** Pandas DataFrame (mixed types).
* **Logic:**
* **Categorical:** Do NOT use One-Hot. Use `nn.Embedding(cardinality, d_model)`.
* **Numerical:** Apply `sklearn.preprocessing.QuantileTransformer(output_distribution='normal')`, then project via `nn.Linear(1, d_model)`.


* **Output:** A tensor of shape `[Batch, Num_Cols, d_model]`.
* **Inverse:** Must perfectly reconstruct the dataframe from tensors.

### B. The Novel Backbone (`dual_stream.py`)

* **Class:** `DualStreamTransformer(nn.Module)`
* **Input:** - `local_tokens`: The column embeddings `[B, N, D]`.
* `global_token`: A learnable parameter `[1, 1, D]` expanded to batch size.
* `time_emb`: Sinusoidal embedding of time .


* **Block Logic (Repeat L times):**
1. **Global-to-Local:** `global_token` cross-attends to `local_tokens` (Updates Global).
2. **Local-to-Global:** `local_tokens` cross-attend to `global_token` (Updates Local).
3. **Local-Self:** `local_tokens` self-attend (Standard SA).
4. **FeedForward:** Standard MLP for both streams.


* **Output:** The final `local_tokens` projected back to input dimension (predicting velocity).

### C. The Engine (`flow_matching.py`)

* **Objective:** Optimal Transport CFM.
* Source: 
* Target:  (Data)
* Interpolation: 
* Loss: MSE between model output  and target velocity .


* **Sampling:** Implement a simple Euler ODE solver (steps=20).
* 



### D. Evaluation Suite (`metrics.py`)

Do not use weak metrics. Implement the "Gold Standard":

1. **Fidelity:**
* **Alpha-Precision:** (Manifold Quality)
* **Beta-Recall:** (Manifold Coverage)
* **Correlation RMSE:** Frobenius norm diff between Real/Syn correlation matrices.


2. **Utility:** - **CatBoost Score:** Train CatBoost on Synthetic, Test on Real (F1 for Class / R2 for Reg).
* **XGBoost Score:** Same as above.


3. **Privacy:**
* **DCR:** Euclidean Distance to Closest Record (on normalized data).

4. **Natural Language Relationship Description:**
* **Core Idea:** Automatically describe relationships between columns in natural language.
* **Method:**
    *   **Numerical Correlations:** Compute correlation matrix. Describe strong positive/negative correlations (> 0.7).
    *   **Categorical-Categorical:** Use contingency tables to identify strong conditional probabilities (P(B|A) > 0.8).
    *   **Categorical-Numerical:** Analyze group statistics (mean/std) to identify significant differences across categories.
*   **Output:** A readable text report summarizing these findings.


## 3. Comparison Pipeline (`run_benchmark.py`)

Create a script that seamlessly integrates with the existing benchmark framework:

1.  **Load Constraints**: Read `../Project/sd_framework/benchmark_results_20260202/adult_constraints.json` to identify target subgroups/constraints (mapped by `hash`).
2.  **Locate Baselines**: For each constraint hash, find existing samples in `../Project/sd_framework/benchmark_results_20260202/data/adult/single/`:
    *   `cuts_adult_{hash}_*.csv`
    *   `tabdiff_adult_{hash}_*.csv`
    *   `relddpm-combined_adult_{hash}_*.csv`
3.  **Generate/Load DS-Flow**:
    *   Generate samples for the same constraints using `DS-Flow`.
    *   Save output using the same naming convention: `dsflow_adult_{hash}_{date}_{count}.csv`.
4.  **Compute Metrics**: Run the `metrics.py` suite on all methods for each constraint.
5.  **Aggregated Reporting**: Output a DataFrame table: `[Method, Constraint_Hash, Alpha, Beta, CatBoost-F1, DCR]`.

# Coding Standards

* Use `torch`, `pandas`, `numpy`, `sklearn`, `catboost`, `xgboost`.
* Use Type Hinting (e.g., `def forward(self, x: torch.Tensor) -> torch.Tensor:`).
* Keep code concise but fully functional.
* Prioritize vectorization over loops.

**Task:** Generate the code for `data/tokenizer.py`, `models/dual_stream.py`, `models/flow_matching.py`, and `evaluation/metrics.py` first.

```