"""
Simple script to run training and generation using config.py
This makes it easy to experiment with different configurations
"""

import argparse
import torch
from train import train_diffusion_model
from generate_test import SyntheticDataGenerator, SyntheticDataEvaluator
import pandas as pd
from config import get_config


def train_model(preset=None):
    """Train the diffusion model using configuration"""
    print("="*80)
    print("TRAINING DIFFUSION TRANSFORMER")
    print("="*80)
    
    # Get configuration
    config = get_config(preset)
    
    print(f"\nUsing preset: {preset if preset else 'default'}")
    print(f"Data path: {config['data']['data_path']}")
    print(f"Hidden dim: {config['model']['hidden_dim']}")
    print(f"Number of layers: {config['model']['num_layers']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Number of epochs: {config['training']['num_epochs']}")
    
    # Check device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("\nWarning: CUDA not available, using CPU instead")
        device = 'cpu'
    
    # Train model
    model, diffusion, preprocessor = train_diffusion_model(
        data_path=config['data']['data_path'],
        categorical_cols=config['data']['categorical_cols'],
        condition_cols=config['data']['condition_cols'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        num_timesteps=config['model']['num_timesteps'],
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        val_split=config['data']['val_split'],
        device=device,
        save_dir=config['training']['save_dir']
    )
    
    print("\n✓ Training completed!")
    print(f"Model saved to: {config['training']['save_dir']}")
    
    return config


def generate_data(config=None, preset=None):
    """Generate synthetic data using trained model"""
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC DATA")
    print("="*80)
    
    if config is None:
        config = get_config(preset)
    
    # Check device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Load generator
    generator = SyntheticDataGenerator(
        model_path=f"{config['training']['save_dir']}/checkpoint_epoch_final.pt",
        preprocessor_path=f"{config['training']['save_dir']}/preprocessor.json",
        device=device
    )
    
    # Generate samples
    num_samples = config['generation']['num_samples']
    print(f"\nGenerating {num_samples} synthetic samples...")
    
    synthetic_df = generator.generate(
        num_samples=num_samples,
        batch_size=config['generation']['batch_size']
    )
    
    # Save synthetic data
    output_path = 'synthetic_data.csv'
    synthetic_df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(synthetic_df)} samples")
    print(f"Synthetic data saved to: {output_path}")
    print(f"\nFirst 5 samples:")
    print(synthetic_df.head())
    
    return synthetic_df


def evaluate_data(config=None, preset=None):
    """Evaluate synthetic data quality"""
    print("\n" + "="*80)
    print("EVALUATING SYNTHETIC DATA")
    print("="*80)
    
    if config is None:
        config = get_config(preset)
    
    # Load data
    print("\nLoading data...")
    real_data = pd.read_csv(config['data']['data_path'])
    synthetic_data = pd.read_csv('synthetic_data.csv')
    
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    
    # Create evaluator
    evaluator = SyntheticDataEvaluator(real_data, synthetic_data)
    
    # Generate comprehensive report
    print("\nGenerating evaluation report...")
    results = evaluator.generate_report(save_dir=config['evaluation']['save_dir'])
    
    # Test ML efficacy if target column is specified
    if config['evaluation']['target_column'] in real_data.columns:
        print("\nTesting ML efficacy...")
        ml_results = evaluator.machine_learning_efficacy(
            target_column=config['evaluation']['target_column'],
            task_type=config['evaluation']['task_type']
        )
        
        print(f"\n✓ Evaluation completed!")
        print(f"Results saved to: {config['evaluation']['save_dir']}")
        
        return results, ml_results
    else:
        print(f"\n✓ Evaluation completed!")
        print(f"Results saved to: {config['evaluation']['save_dir']}")
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Train and generate synthetic tabular data using Diffusion Transformer'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'generate', 'evaluate', 'full'],
        help='Operation mode: train, generate, evaluate, or full pipeline'
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        default=None,
        choices=['small', 'medium', 'large', 'high_quality', 'fast'],
        help='Configuration preset to use'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to data CSV file (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Get config and override data path if provided
    config = get_config(args.preset)
    if args.data:
        config['data']['data_path'] = args.data
    
    # Execute requested mode
    if args.mode == 'train':
        train_model(args.preset)
        
    elif args.mode == 'generate':
        generate_data(config, args.preset)
        
    elif args.mode == 'evaluate':
        evaluate_data(config, args.preset)
        
    elif args.mode == 'full':
        # Run full pipeline
        print("Running full pipeline: Train -> Generate -> Evaluate\n")
        
        # Train
        config = train_model(args.preset)
        
        # Generate
        synthetic_df = generate_data(config, args.preset)
        
        # Evaluate
        evaluate_data(config, args.preset)
        
        print("\n" + "="*80)
        print("FULL PIPELINE COMPLETED!")
        print("="*80)
        print("\nGenerated files:")
        print(f"  - {config['training']['save_dir']}/: Model checkpoints")
        print(f"  - synthetic_data.csv: Generated synthetic data")
        print(f"  - {config['evaluation']['save_dir']}/: Evaluation results")


if __name__ == '__main__':
    main()
