"""
Complete Example: Training and Testing Diffusion Transformer on Tabular Data
This script demonstrates the full workflow from data preparation to evaluation
"""

import torch
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

from train import train_diffusion_model
from generate_test import SyntheticDataGenerator, SyntheticDataEvaluator


def create_sample_dataset(n_samples=5000, dataset_type='classification'):
    """
    Create a sample tabular dataset for demonstration
    
    Args:
        n_samples: Number of samples
        dataset_type: 'classification' or 'regression'
    
    Returns:
        DataFrame with sample data
    """
    print(f"Creating sample {dataset_type} dataset...")
    
    if dataset_type == 'classification':
        # Create classification dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            n_classes=3,
            random_state=42
        )
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        
        # Add target
        df['target'] = y
        
        # Add categorical features
        df['category_1'] = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
        df['category_2'] = np.random.choice(['X', 'Y', 'Z'], size=n_samples)
        
        # Add a condition column (e.g., age group)
        df['age_group'] = np.random.choice(['young', 'middle', 'senior'], size=n_samples)
        
    else:
        # Create regression dataset
        X, y = make_regression(
            n_samples=n_samples,
            n_features=10,
            n_informative=7,
            noise=0.1,
            random_state=42
        )
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        
        # Add target
        df['target'] = y
        
        # Add categorical features
        df['category_1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
        df['category_2'] = np.random.choice(['X', 'Y'], size=n_samples)
        
        # Add a condition column
        df['region'] = np.random.choice(['north', 'south', 'east', 'west'], size=n_samples)
    
    return df


def example_unconditional_generation():
    """
    Example 1: Unconditional generation
    Generate synthetic data without any conditions
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: UNCONDITIONAL GENERATION")
    print("="*80)
    
    # Create sample dataset
    df = create_sample_dataset(n_samples=3000, dataset_type='classification')
    df.to_csv('sample_data.csv', index=False)
    print(f"Sample dataset created with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Train the model
    print("\n--- Training Diffusion Model ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, diffusion, preprocessor = train_diffusion_model(
        data_path='sample_data.csv',
        categorical_cols=['category_1', 'category_2', 'age_group'],
        condition_cols=None,  # No conditioning
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        num_timesteps=1000,
        batch_size=128,
        num_epochs=50,
        learning_rate=1e-4,
        val_split=0.1,
        device=device,
        save_dir='checkpoints_unconditional'
    )
    
    # Generate synthetic data
    print("\n--- Generating Synthetic Data ---")
    generator = SyntheticDataGenerator(
        model_path='checkpoints_unconditional/checkpoint_epoch_final.pt',
        preprocessor_path='checkpoints_unconditional/preprocessor.json',
        device=device
    )
    
    synthetic_df = generator.generate(num_samples=1000)
    synthetic_df.to_csv('synthetic_data_unconditional.csv', index=False)
    print(f"Generated {len(synthetic_df)} synthetic samples")
    print("\nFirst 5 synthetic samples:")
    print(synthetic_df.head())
    
    # Evaluate synthetic data
    print("\n--- Evaluating Synthetic Data ---")
    evaluator = SyntheticDataEvaluator(df, synthetic_df)
    results = evaluator.generate_report(save_dir='evaluation_unconditional')
    
    # Test ML efficacy
    ml_results = evaluator.machine_learning_efficacy(
        target_column='target',
        task_type='classification'
    )
    
    print("\n✓ Unconditional generation example completed!")


def example_conditional_generation():
    """
    Example 2: Conditional generation
    Generate synthetic data conditioned on specific features
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: CONDITIONAL GENERATION")
    print("="*80)
    
    # Create sample dataset
    df = create_sample_dataset(n_samples=3000, dataset_type='classification')
    df.to_csv('sample_data_conditional.csv', index=False)
    print(f"Sample dataset created with shape: {df.shape}")
    
    # Train the model with conditioning
    print("\n--- Training Conditional Diffusion Model ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, diffusion, preprocessor = train_diffusion_model(
        data_path='sample_data_conditional.csv',
        categorical_cols=['category_1', 'category_2', 'age_group'],
        condition_cols=['age_group'],  # Condition on age group
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        num_timesteps=1000,
        batch_size=128,
        num_epochs=50,
        learning_rate=1e-4,
        val_split=0.1,
        device=device,
        save_dir='checkpoints_conditional'
    )
    
    # Generate conditional synthetic data
    print("\n--- Generating Conditional Synthetic Data ---")
    generator = SyntheticDataGenerator(
        model_path='checkpoints_conditional/checkpoint_epoch_final.pt',
        preprocessor_path='checkpoints_conditional/preprocessor.json',
        device=device
    )
    
    # Generate samples for each age group
    all_synthetic = []
    for age_group in ['young', 'middle', 'senior']:
        print(f"\nGenerating samples for age_group: {age_group}")
        condition_values = pd.DataFrame({'age_group': [age_group]})
        synthetic_df = generator.generate_conditional(
            condition_values=condition_values,
            num_samples_per_condition=300
        )
        all_synthetic.append(synthetic_df)
        
        print(f"Generated {len(synthetic_df)} samples for {age_group}")
        print(f"Age group distribution: {synthetic_df['age_group'].value_counts().to_dict()}")
    
    # Combine all conditional samples
    synthetic_df_all = pd.concat(all_synthetic, ignore_index=True)
    synthetic_df_all.to_csv('synthetic_data_conditional.csv', index=False)
    
    print(f"\nTotal synthetic samples: {len(synthetic_df_all)}")
    print("Age group distribution in synthetic data:")
    print(synthetic_df_all['age_group'].value_counts())
    
    # Evaluate synthetic data
    print("\n--- Evaluating Conditional Synthetic Data ---")
    evaluator = SyntheticDataEvaluator(df, synthetic_df_all)
    results = evaluator.generate_report(save_dir='evaluation_conditional')
    
    # Test ML efficacy
    ml_results = evaluator.machine_learning_efficacy(
        target_column='target',
        task_type='classification'
    )
    
    print("\n✓ Conditional generation example completed!")


def example_custom_dataset():
    """
    Example 3: Using your own dataset
    Template for using custom data
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: USING CUSTOM DATASET (Template)")
    print("="*80)
    
    print("""
    To use your own dataset, follow these steps:
    
    1. Prepare your CSV file with tabular data
    
    2. Identify categorical columns in your data
    
    3. (Optional) Identify columns to use as conditions
    
    4. Train the model:
    
    from train import train_diffusion_model
    
    model, diffusion, preprocessor = train_diffusion_model(
        data_path='your_data.csv',
        categorical_cols=['col1', 'col2', 'col3'],  # List your categorical columns
        condition_cols=['condition_col'],            # Optional: conditioning columns
        hidden_dim=256,                              # Adjust based on data complexity
        num_layers=6,                                # More layers = more capacity
        num_heads=8,                                 # Number of attention heads
        num_timesteps=1000,                          # Diffusion steps
        batch_size=128,                              # Adjust based on GPU memory
        num_epochs=100,                              # Train longer for better quality
        learning_rate=1e-4,
        val_split=0.1,
        device='cuda',
        save_dir='my_checkpoints'
    )
    
    5. Generate synthetic data:
    
    from generate_test import SyntheticDataGenerator
    
    generator = SyntheticDataGenerator(
        model_path='my_checkpoints/checkpoint_epoch_final.pt',
        preprocessor_path='my_checkpoints/preprocessor.json',
        device='cuda'
    )
    
    synthetic_df = generator.generate(num_samples=5000)
    synthetic_df.to_csv('my_synthetic_data.csv', index=False)
    
    6. Evaluate the quality:
    
    from generate_test import SyntheticDataEvaluator
    import pandas as pd
    
    real_df = pd.read_csv('your_data.csv')
    synthetic_df = pd.read_csv('my_synthetic_data.csv')
    
    evaluator = SyntheticDataEvaluator(real_df, synthetic_df)
    evaluator.generate_report(save_dir='my_evaluation')
    evaluator.machine_learning_efficacy(target_column='target', task_type='classification')
    """)


def example_hyperparameter_tuning():
    """
    Example 4: Hyperparameter recommendations
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: HYPERPARAMETER TUNING GUIDE")
    print("="*80)
    
    print("""
    Hyperparameter recommendations based on dataset size and complexity:
    
    SMALL DATASETS (< 1,000 samples):
    - hidden_dim: 64-128
    - num_layers: 2-4
    - num_heads: 2-4
    - batch_size: 32-64
    - num_epochs: 100-200
    - num_timesteps: 500-1000
    
    MEDIUM DATASETS (1,000-10,000 samples):
    - hidden_dim: 128-256
    - num_layers: 4-6
    - num_heads: 4-8
    - batch_size: 128-256
    - num_epochs: 50-100
    - num_timesteps: 1000
    
    LARGE DATASETS (> 10,000 samples):
    - hidden_dim: 256-512
    - num_layers: 6-8
    - num_heads: 8-16
    - batch_size: 256-512
    - num_epochs: 30-50
    - num_timesteps: 1000
    
    TIPS:
    - Increase hidden_dim and num_layers for more complex patterns
    - Use more epochs for smaller datasets
    - Adjust batch_size based on GPU memory
    - Use learning rate 1e-4 as default, try 5e-5 or 2e-4 if needed
    - Monitor validation loss to avoid overfitting
    - Use cosine schedule for smoother training
    """)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Diffusion Transformer Examples')
    parser.add_argument('--example', type=str, default='all',
                       choices=['all', '1', '2', '3', '4'],
                       help='Which example to run (1-4 or all)')
    args = parser.parse_args()
    
    if args.example == 'all' or args.example == '1':
        example_unconditional_generation()
    
    if args.example == 'all' or args.example == '2':
        example_conditional_generation()
    
    if args.example == '3':
        example_custom_dataset()
    
    if args.example == '4':
        example_hyperparameter_tuning()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  - sample_data.csv / sample_data_conditional.csv")
    print("  - synthetic_data_unconditional.csv / synthetic_data_conditional.csv")
    print("  - checkpoints_*/: Model checkpoints")
    print("  - evaluation_*/: Evaluation reports and plots")
    print("\nFor custom datasets, see Example 3 template above.")
