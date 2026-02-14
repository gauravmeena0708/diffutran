# Quick Start Guide

This guide will help you get started with the Diffusion Transformer for tabular data generation in just a few minutes!

## Installation (1 minute)

```bash
# Install required packages
pip install -r requirements.txt
```

## Three Ways to Use This Code

### Option 1: Using the Simple Run Script (Easiest! ⭐)

1. **Edit the configuration file (`config.py`)**:
   ```python
   DATA_CONFIG = {
       'data_path': 'your_data.csv',  # Change this to your CSV file
       'categorical_cols': ['col1', 'col2'],  # List your categorical columns
       'condition_cols': None,  # or ['condition_col'] for conditional generation
   }
   ```

2. **Run the full pipeline**:
   ```bash
   # Train, generate, and evaluate in one command!
   python run.py --mode full --preset medium
   
   # Or run steps individually:
   python run.py --mode train --preset medium
   python run.py --mode generate
   python run.py --mode evaluate
   ```

**Presets available:**
- `small`: For datasets < 1,000 samples
- `medium`: For datasets 1,000-10,000 samples (default)
- `large`: For datasets > 10,000 samples
- `high_quality`: Best quality (slower)
- `fast`: Quick training (lower quality)

### Option 2: Using Python API (Flexible)

```python
from train import train_diffusion_model
from generate_test import SyntheticDataGenerator

# 1. Train
model, diffusion, preprocessor = train_diffusion_model(
    data_path='your_data.csv',
    categorical_cols=['cat1', 'cat2'],
    condition_cols=None,  # Optional: ['condition_col']
    hidden_dim=256,
    num_layers=6,
    num_epochs=100,
    save_dir='checkpoints'
)

# 2. Generate
generator = SyntheticDataGenerator(
    model_path='checkpoints/checkpoint_epoch_final.pt',
    preprocessor_path='checkpoints/preprocessor.json'
)

synthetic_data = generator.generate(num_samples=5000)
synthetic_data.to_csv('synthetic_data.csv', index=False)
```

### Option 3: Using Complete Examples (Learning)

```bash
# Run example demonstrations
python example_usage.py --example 1  # Unconditional generation
python example_usage.py --example 2  # Conditional generation
python example_usage.py --example all  # All examples
```

## Your First Synthetic Dataset (5 minutes)

Let's create a sample dataset and generate synthetic data:

```bash
# 1. Run the complete example (creates sample data, trains, and generates)
python example_usage.py --example 1

# This will:
#   - Create sample_data.csv
#   - Train a diffusion model
#   - Generate synthetic_data_unconditional.csv
#   - Evaluate the quality
#   - Create evaluation plots
```

## Understanding Your Results

After running, you'll get:

1. **Checkpoints**: Trained model saved in `checkpoints/` directory
   - `checkpoint_epoch_final.pt`: Final trained model
   - `preprocessor.json`: Data preprocessing info
   - `losses.png`: Training progress plot

2. **Synthetic Data**: `synthetic_data.csv` with generated samples

3. **Evaluation**: `evaluation_results/` directory with:
   - `evaluation_report.json`: Statistical comparison
   - `correlation_comparison.png`: Correlation matrices
   - ML efficacy scores (TRTR vs TSTR)

## Common Use Cases

### Generate More Data for Machine Learning

```python
from generate_test import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    model_path='checkpoints/checkpoint_epoch_final.pt',
    preprocessor_path='checkpoints/preprocessor.json'
)

# Generate 10,000 new training samples
synthetic_data = generator.generate(num_samples=10000)
```

### Generate Data for Specific Conditions

```python
import pandas as pd

# Generate samples for specific groups
condition_values = pd.DataFrame({
    'region': ['north', 'south', 'east', 'west'],
})

synthetic_data = generator.generate_conditional(
    condition_values=condition_values,
    num_samples_per_condition=1000
)
```

### Balance Your Dataset

```python
# Generate more samples for underrepresented classes
minority_conditions = pd.DataFrame({
    'class': ['rare_class'] * 1000
})

balanced_data = generator.generate_conditional(
    condition_values=minority_conditions,
    num_samples_per_condition=1
)
```

## Tips for Best Results

1. **Data Preparation**:
   - Remove or handle missing values before training
   - Ensure categorical columns are properly specified
   - Use enough samples (>1000 recommended)

2. **Model Size**:
   - Start with default settings (medium preset)
   - Increase model size if results aren't good enough
   - Decrease if training is too slow

3. **Training**:
   - Monitor the loss curve (`checkpoints/losses.png`)
   - If loss plateaus early, try a larger model
   - If loss is unstable, reduce learning rate

4. **Evaluation**:
   - Check statistical similarity (means, stds, distributions)
   - Look at correlation preservation
   - Test ML efficacy (TSTR/TRTR ratio should be > 0.8)

## Troubleshooting

**"Out of memory" error:**
```python
# Reduce batch size in config.py
TRAINING_CONFIG = {
    'batch_size': 32,  # Reduce from 128
}
```

**Poor quality synthetic data:**
```bash
# Use high quality preset
python run.py --mode full --preset high_quality
```

**Training too slow:**
```bash
# Use fast preset
python run.py --mode full --preset fast
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [example_usage.py](example_usage.py) for more examples
- Modify [config.py](config.py) to customize settings
- Explore the code in `diffusion_model.py`, `train.py`, and `generate_test.py`

## Need Help?

- Check if your CSV file is properly formatted
- Ensure categorical columns are correctly specified
- Try different presets (small, medium, large)
- Look at the evaluation plots to understand quality
- Adjust hyperparameters based on your dataset size

---

**Happy Generating! 🎉**

If you have questions, refer to the main [README.md](README.md) or open an issue.
