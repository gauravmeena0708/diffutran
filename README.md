# DiffuTran: Models for Tabular Data Generation

This repository provides two deep learning models for generating synthetic tabular data:
1.  **DiffuTron**: A Transformer-based Denoising Diffusion Probabilistic Model (DDPM).
2.  **DS-Flow**: A Dual-Stream Transformer model with Flow Matching.

These models are designed to learn the underlying distribution of a given tabular dataset and generate new, synthetic samples that preserve the statistical properties of the original data.

## Features

-   **State-of-the-Art DS-Flow**: Upgraded Dual-Stream Flow model featuring **FlashAttention-2** and **1D-RoPE** for highly scalable and memory-efficient tabular processing.
-   **Mixed Continuous-Discrete Flow Matching**: Numericals generated via continuous Optimal Transport Flow Matching (OT-CFM), while Categoricals utilize discrete Classifier-Free DDPM transitions for rigorous discrete synthesis.
-   **Advanced Solvers**: Utilizes **DPM-Solver++** combined with adaptive CFG decay to achieve high-fidelity generation in <10 steps.
-   **Robust Data Layer**: Native Missing Value tracking, zero-memory high-cardinality hashing fallbacks (>100 categories), and robust numerical outlier jitter scaling.
-   **Comprehensive Evaluation Framework**: Includes rigorous benchmarking:
    -   **Fidelity**: Validated via exact Kolmogorov-Smirnov (KS) tests and strict Chi-Square tables.
    -   **Utility**: ML efficacy (`TSTR` vs `TRTR`) and Discriminator Indistinguishability AUC.
    -   **Privacy**: Distance to Closest Record (DCR), k-Anonymity density checks, l-Diversity bounds, and Membership Inference Attack (MIA) capability testing.

## Project Structure

```
/mnt/c/Users/gaura/Documents/GitHub/diffutran/
├───data/                 # Sample datasets
├───scripts/              # High-level scripts for training, generation, and evaluation
├───src/
│   ├───common/           # Shared utilities (data preprocessing, evaluation metrics)
│   ├───diffutron/        # Implementation of the Diffusion Transformer model
│   └───flow_model/       # Implementation of the Dual-Stream Flow model
└───tests/                # Unit tests
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/diffutran.git
    cd diffutran
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This project provides separate scripts to train and use each model.

---

### 1. DiffuTron (Diffusion Transformer)

#### Training

To train the DiffuTron model, use the `scripts/train.py` script.

```bash
python scripts/train.py 
  --data_path data/datasets/adult/train.csv 
  --categorical_cols workclass education marital-status occupation relationship race sex native-country 
  --condition_col income 
  --hidden_dim 512 
  --num_layers 6 
  --num_heads 8 
  --num_epochs 100 
  --batch_size 512 
  --save_dir outputs/checkpoints_adult_diffutron
```

**Key Arguments:**
-   `--data_path`: Path to your training CSV file.
-   `--categorical_cols`: A list of column names that should be treated as categorical.
-   `--condition_col`: The column to use for conditional generation.
-   `--save_dir`: Directory to save model checkpoints and the data preprocessor.

#### Generation

To generate synthetic data using a trained DiffuTron model, use the `scripts/generate_test.py` script.

```bash
python scripts/generate_test.py 
  --checkpoint_path outputs/checkpoints_adult_diffutron/checkpoint_epoch_final.pt 
  --preprocessor_path outputs/checkpoints_adult_diffutron/preprocessor.json 
  --condition_encoder_path outputs/checkpoints_adult_diffutron/condition_encoder.joblib 
  --real_data_path data/datasets/adult/test.csv 
  --output_dir outputs/synthetic_data/diffutron 
  --num_samples 2000 
  --condition_col income
```

---

### 2. DS-Flow (Dual-Stream Flow Model)

#### Training

To train the DS-Flow model, use the `src/flow_model/train.py` script.

```bash
python src/flow_model/train.py 
  --data_path data/datasets/adult/train.csv 
  --categorical_cols workclass education marital-status occupation relationship race sex native-country 
  --condition_col income 
  --num_epochs 100 
  --batch_size 512 
  --d_model 512 
  --save_dir outputs/checkpoints_adult_dsflow
```

**Key Arguments:**
-   `--data_path`: Path to your training CSV file.
-   `--categorical_cols`: A list of categorical column names.
-   `--condition_col`: The column to use for conditional generation.
-   `--save_dir`: Directory to save model checkpoints and the tokenizer.

#### Generation

To generate data with a trained DS-Flow model, use the `src/flow_model/sampler.py` script.

```bash
python src/flow_model/sampler.py 
  --checkpoint_path outputs/checkpoints_adult_dsflow/model_final.pt 
  --tokenizer_path outputs/checkpoints_adult_dsflow/tokenizer.pt 
  --output_dir outputs/synthetic_data/dsflow 
  --num_samples 2000 
  --condition_col income
```

---

### Evaluation

The `scripts/run_benchmark.py` script provides a comprehensive framework for evaluating and comparing the generative models. It loads trained models, generates data for specific constraints, and computes a suite of fidelity, utility, and privacy metrics.

```bash
python scripts/run_benchmark.py 
  --ds_checkpoint <path_to_ds_flow_checkpoint> 
  --ds_tokenizer <path_to_ds_flow_tokenizer> 
  --m1_checkpoint <path_to_diffutron_checkpoint> 
  --m1_preprocessor <path_to_diffutron_preprocessor> 
  --m1_condition_encoder <path_to_diffutron_condition_encoder>
```

The script will generate a `benchmark_comparison.csv` file in the `outputs/benchmark_results` directory with the detailed metrics.

## Model Architectures

### DiffuTron

This model is based on Denoising Diffusion Probabilistic Models (DDPM). It uses a Transformer architecture to predict the noise that was added to the tabular data at a given timestep. The training objective is to reconstruct the original data by iteratively denoising from a pure noise vector. This model supports Classifier-Free Guidance (CFG) for improved conditional generation.

### DS-Flow (v2 SOTA)

This model utilizes an advanced hybrid of Continuous Flow Matching and Discrete DDPM. It features a novel **Dual-Stream Transformer** customized with **FlashAttention-2**, **1D Rotary Positional Embeddings (RoPE)**, and **Dedicated Cross-Attention Conditioning**. It processes numerical features through continuous Optimal Transport limit paths and categorical features via principled discrete probability reverse chains. Paired with an intertwined adaptive **DPM-Solver++**, it achieves state-of-the-art data synthesis speed and fidelity while preventing mode-collapse.
