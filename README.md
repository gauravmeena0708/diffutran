# Diffusion Transformer for Tabular Data Generation

A PyTorch implementation of a Diffusion Transformer model for generating high-quality synthetic conditional tabular datasets. This implementation combines the power of Transformer architecture with Denoising Diffusion Probabilistic Models (DDPM) for tabular data synthesis.

## Features

- ✨ **Transformer-based Architecture**: Utilizes multi-head self-attention for capturing complex feature relationships
- 🎯 **Conditional Generation**: Generate synthetic data conditioned on specific features
- 🔄 **Flexible Preprocessing**: Automatic handling of both numerical and categorical features
- 📊 **Comprehensive Evaluation**: Built-in metrics for assessing synthetic data quality
- 🚀 **Easy to Use**: Simple API for training and generation
- 💾 **State Management**: Save and load trained models with preprocessors

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Examples](#examples)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Advanced Features](#advanced-features)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for faster training)

### Setup

```bash
# Clone or download the repository
cd diffusion-transformer-tabular

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Your data should be in CSV format with both numerical and categorical columns:

```python
import pandas as pd

# Example dataset structure
df = pd.read_csv('your_data.csv')
print(df.head())
#    feature_1  feature_2  category_1  category_2  target
# 0       1.23       4.56           A           X       0
# 1       2.34       5.67           B           Y       1
# ...
```

### 2. Train the Model

```python
from train import train_diffusion_model

model, diffusion, preprocessor = train_diffusion_model(
    data_path='your_data.csv',
    categorical_cols=['category_1', 'category_2'],  # Specify categorical columns
    condition_cols=None,                             # None for unconditional generation
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    num_timesteps=1000,
    batch_size=128,
    num_epochs=100,
    learning_rate=1e-4,
    val_split=0.1,
    device='cuda',
    save_dir='checkpoints'
)
```

### 3. Generate Synthetic Data

```python
from generate_test import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    model_path='checkpoints/checkpoint_epoch_final.pt',
    preprocessor_path='checkpoints/preprocessor.json',
    device='cuda'
)

# Generate 1000 synthetic samples
synthetic_df = generator.generate(num_samples=1000)
synthetic_df.to_csv('synthetic_data.csv', index=False)
```

### 4. Evaluate Quality

```python
from generate_test import SyntheticDataEvaluator

evaluator = SyntheticDataEvaluator(real_data, synthetic_data)
evaluator.generate_report(save_dir='evaluation_results')
evaluator.machine_learning_efficacy(target_column='target', task_type='classification')
```

## Project Structure

```
.
├── diffusion_model.py      # Core model architecture
├── train.py                # Training script and utilities
├── generate_test.py        # Generation and evaluation
├── example_usage.py        # Complete examples
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### File Descriptions

- **diffusion_model.py**: Contains the DiffusionTransformer model and GaussianDiffusion process
- **train.py**: Training pipeline, data preprocessing, and trainer class
- **generate_test.py**: Synthetic data generation and comprehensive evaluation metrics
- **example_usage.py**: Complete working examples with different scenarios

## Usage Guide

### Unconditional Generation

Generate synthetic data without conditioning:

```python
from train import train_diffusion_model
from generate_test import SyntheticDataGenerator

# Train
train_diffusion_model(
    data_path='data.csv',
    categorical_cols=['cat1', 'cat2'],
    condition_cols=None,  # No conditioning
    num_epochs=100,
    save_dir='checkpoints_unconditional'
)

# Generate
generator = SyntheticDataGenerator(
    model_path='checkpoints_unconditional/checkpoint_epoch_final.pt',
    preprocessor_path='checkpoints_unconditional/preprocessor.json'
)

synthetic_data = generator.generate(num_samples=5000)
```

### Conditional Generation

Generate synthetic data conditioned on specific features:

```python
# Train with conditioning
train_diffusion_model(
    data_path='data.csv',
    categorical_cols=['cat1', 'cat2'],
    condition_cols=['region', 'age_group'],  # Condition on these columns
    num_epochs=100,
    save_dir='checkpoints_conditional'
)

# Generate for specific conditions
generator = SyntheticDataGenerator(
    model_path='checkpoints_conditional/checkpoint_epoch_final.pt',
    preprocessor_path='checkpoints_conditional/preprocessor.json'
)

# Generate 500 samples for each region
condition_values = pd.DataFrame({'region': ['north', 'south', 'east', 'west']})
synthetic_data = generator.generate_conditional(
    condition_values=condition_values,
    num_samples_per_condition=500
)
```

## Model Architecture

### Diffusion Transformer

The model consists of:

1. **Time Embedding**: Sinusoidal positional encoding for timestep information
2. **Input Projection**: Projects input features to hidden dimension
3. **Condition Projection**: (Optional) Projects conditioning features
4. **Transformer Blocks**: Multiple layers of self-attention and feed-forward networks
5. **Adaptive Layer Norm**: Time-conditioned normalization
6. **Output Projection**: Projects back to original feature dimension

### Key Components

```python
DiffusionTransformer(
    input_dim=10,           # Number of features to generate
    hidden_dim=256,         # Hidden dimension of transformer
    num_layers=6,           # Number of transformer blocks
    num_heads=8,            # Number of attention heads
    mlp_ratio=4.0,          # MLP expansion ratio
    dropout=0.1,            # Dropout rate
    time_embed_dim=128,     # Time embedding dimension
    condition_dim=None      # Conditioning dimension (None if unconditional)
)
```

### Diffusion Process

The model uses a Gaussian diffusion process with:

- **Forward Process**: Gradually adds noise to data over T timesteps
- **Reverse Process**: Learns to denoise data step by step
- **Training Objective**: Predict the noise added at each timestep
- **Sampling**: Start from pure noise and iteratively denoise

## Evaluation Metrics

The evaluation suite includes:

### 1. Statistical Similarity

Compares distributions between real and synthetic data:

- **Numerical Features**: Mean, standard deviation, and their differences
- **Categorical Features**: Jensen-Shannon divergence between distributions

### 2. Correlation Analysis

- Correlation matrix comparison
- Frobenius norm of correlation differences
- Visual heatmap comparisons

### 3. Machine Learning Efficacy

Tests if synthetic data is useful for ML tasks:

- **TRTR**: Train on Real, Test on Real (baseline)
- **TSTR**: Train on Synthetic, Test on Real (target)
- **Metric**: Accuracy (classification) or R² (regression)

Good synthetic data should have TSTR/TRTR ratio close to 1.0.

### 4. Visualization

Generates comparison plots including:
- Correlation heatmaps
- Distribution comparisons
- Training loss curves

## Examples

Run the complete examples:

```bash
# Run all examples
python example_usage.py --example all

# Run specific example
python example_usage.py --example 1  # Unconditional generation
python example_usage.py --example 2  # Conditional generation
python example_usage.py --example 3  # Custom dataset template
python example_usage.py --example 4  # Hyperparameter guide
```

## Hyperparameter Tuning

### Small Datasets (< 1,000 samples)

```python
train_diffusion_model(
    data_path='small_data.csv',
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    batch_size=32,
    num_epochs=200,
    num_timesteps=500
)
```

### Medium Datasets (1,000-10,000 samples)

```python
train_diffusion_model(
    data_path='medium_data.csv',
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    batch_size=128,
    num_epochs=100,
    num_timesteps=1000
)
```

### Large Datasets (> 10,000 samples)

```python
train_diffusion_model(
    data_path='large_data.csv',
    hidden_dim=512,
    num_layers=8,
    num_heads=16,
    batch_size=256,
    num_epochs=50,
    num_timesteps=1000
)
```

### General Tips

- **Hidden Dimension**: Increase for more complex patterns (64-512)
- **Number of Layers**: More layers = more capacity (2-8)
- **Attention Heads**: Typically 4-16 heads work well
- **Batch Size**: Larger is better (limited by GPU memory)
- **Learning Rate**: Start with 1e-4, adjust if needed (5e-5 to 2e-4)
- **Epochs**: More epochs for smaller datasets
- **Timesteps**: 1000 is standard, 500 can work for simpler data

## Advanced Features

### Custom Loss Functions

Modify the loss in `diffusion_model.py`:

```python
def p_losses(self, model, x_start, t, condition=None, noise=None):
    # Add custom loss components here
    loss = F.mse_loss(predicted_noise, noise)
    # loss += custom_regularization
    return loss
```

### Data Augmentation

Add augmentation in `train.py`:

```python
class TabularDataset(Dataset):
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Add augmentation here
        return sample
```

### Custom Evaluation Metrics

Extend the evaluator in `generate_test.py`:

```python
class SyntheticDataEvaluator:
    def custom_metric(self):
        # Implement your custom evaluation
        pass
```

### Resume Training

Load checkpoint and continue training:

```python
trainer = Trainer(...)
trainer.load_checkpoint('checkpoints/checkpoint_epoch_50.pt')
trainer.train(num_epochs=50)  # Train for 50 more epochs
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size`
- Reduce `hidden_dim` or `num_layers`
- Use gradient accumulation

### Poor Generation Quality

- Train for more epochs
- Increase model capacity (`hidden_dim`, `num_layers`)
- Check data preprocessing
- Adjust learning rate
- Use more timesteps

### Training Instability

- Reduce learning rate
- Add gradient clipping (already implemented)
- Use warmup schedule
- Check for data normalization issues

### Slow Training

- Increase batch size (if GPU memory allows)
- Use multiple GPUs (requires code modification)
- Reduce number of timesteps during training
- Use mixed precision training (fp16)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{diffusion_transformer_tabular,
  title={Diffusion Transformer for Tabular Data Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/diffusion-transformer-tabular}
}
```

## References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com]

---

**Happy Synthesizing! 🚀**
