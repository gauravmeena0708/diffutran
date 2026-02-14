import torch
import numpy as np
import pandas as pd
import os
import argparse
import json
import matplotlib.pyplot as plt

# Modular imports
from models.diffusion import DiffusionModel
from models.gaussian_diffusion import GaussianDiffusion
from train import DataPreprocessor
from evaluation.metrics import EvaluationFramework

def load_model(checkpoint_path, preprocessor_path, condition_encoder_path=None, 
               hidden_dim=256, num_layers=6, num_heads=8, device='cuda'):
    """Load trained model and preprocessor"""
    print(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = DataPreprocessor()
    preprocessor.load(preprocessor_path)
    
    # Load condition encoder if exists
    condition_encoder = None
    num_classes = None
    if condition_encoder_path and os.path.exists(condition_encoder_path):
        import joblib
        condition_encoder = joblib.load(condition_encoder_path)
        num_classes = len(condition_encoder.classes_)
    
    # Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Model Config
    num_numerical_features = len(preprocessor.numerical_cols)
    categorical_cardinalities = preprocessor.categorical_cardinalities
    
    model = DiffusionModel(
        num_numerical_features=num_numerical_features,
        categorical_cardinalities=categorical_cardinalities,
        d_model=hidden_dim,
        num_layers=num_layers,
        nhead=num_heads,
        num_classes=num_classes
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, preprocessor, condition_encoder

def generate_data(
    model, 
    diffusion, 
    preprocessor, 
    num_samples=1000, 
    condition=None, # Class index or None
    guidance_scale=1.0, 
    device='cuda'
):
    """
    Generate synthetic data
    Args:
        condition: int (class index) or list of ints. If None, unconditional.
    """
    # Determine shape
    # Total input dim = sum of feature dims (num + one-hots)
    # However, preprocessor doesn't expose total dim directly?
    # We can infer from preprocessor.fit_transform on dummy data or store it.
    
    # Let's calculate total dim
    total_dim = len(preprocessor.numerical_cols) + sum(preprocessor.categorical_cardinalities)
    
    shape = (num_samples, total_dim)
    
    # Prepare conditions
    cond_tensor = None
    if model.num_classes is not None:
        if condition is None:
            # Unconditional generation for conditional model (use null token?)
            # Usually we want to sample from specific classes or marginal.
            # If we pass None to model.forward with conditional model, it uses null token (unconditional output).
            pass # cond_tensor remains None
        else:
            if isinstance(condition, int):
                cond_tensor = torch.full((num_samples,), condition, device=device, dtype=torch.long)
            elif isinstance(condition, (list, np.ndarray)):
                 cond_tensor = torch.tensor(condition, device=device, dtype=torch.long)
    
    # Sample
    print(f"Generating {num_samples} samples with guidance_scale={guidance_scale}...")
    sampled_data = diffusion.sample(
        model, 
        shape, 
        condition=cond_tensor, 
        guidance_scale=guidance_scale, 
        device=device
    )
    
    # Inverse transform
    print("Inverse transforming data...")
    synthetic_df = preprocessor.inverse_transform(sampled_data)
    
    # If we conditioned on a column, we should add it back to the dataframe?
    # The preprocessor.inverse_transform only returns features it managed.
    # If the condition column was dropped during training, we need to add it manually.
    
    return synthetic_df

def main(args):
    device = args.device
    
    # Load resources
    model, preprocessor, condition_encoder = load_model(
        args.checkpoint_path, 
        args.preprocessor_path, 
        args.condition_encoder_path,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        device=device
    )
    
    diffusion = GaussianDiffusion(num_timesteps=1000) # Default
    
    # Real Data for Evaluation
    real_data = pd.read_csv(args.real_data_path)
    
    # Determine generation condition
    # If model is conditional, we can:
    # 1. Generate based on real data distribution of conditions (reproduce real set)
    # 2. Generate balanced classes
    # 3. Generate specific class
    
    conditions = None
    if condition_encoder:
        # Strategy: Reproduce label distribution of real data
        # Check if condition col exists in real data
        # We need the condition column name. It's not stored in preprocessor.
        # It assumes we know it or pass it.
        pass # For now, simple generation
        
        # If we want to evaluate fidelity properly, we should generate with same label distribution as real data
        # assuming we have the label in real_data.
        # Getting label col name is tricky if not passed.
        
        # Fallback: Generate unconditional (cond=None) if user didn't specify?
        # But if we want to evaluate utility, we need labels.
        # If we generate unconditional, we get "null" class samples?
        # No, CFG with cond=None generates "unconditional" distribution, which approximates marginal p(x).
        # But we don't get the labels Y for TSTR.
        
        # BETTER: Generate Y first (sample from prior), then Generate X | Y.
        if args.sample_condition_dist and args.condition_col:
             real_labels = real_data[args.condition_col]
             sampled_labels_str = real_labels.sample(args.num_samples, replace=True).astype(str).values
             conditions = condition_encoder.transform(sampled_labels_str)
        
    synthetics = []
    
    # Generate in batches
    batch_size = args.batch_size
    num_generated = 0
    
    while num_generated < args.num_samples:
        current_batch_size = min(batch_size, args.num_samples - num_generated)
        
        # Prepare batch conditions if needed
        batch_cond = None
        if conditions is not None:
             batch_cond = conditions[num_generated : num_generated + current_batch_size]
        
        syn_batch = generate_data(
            model, diffusion, preprocessor, 
            num_samples=current_batch_size, 
            condition=batch_cond,
            guidance_scale=args.guidance_scale,
            device=device
        )
        
        # Add conditions back to dataframe if using conditioned generation
        if batch_cond is not None and args.condition_col:
            syn_batch[args.condition_col] = condition_encoder.inverse_transform(batch_cond)
            
        synthetics.append(syn_batch)
        num_generated += current_batch_size
        
    synthetic_data = pd.concat(synthetics, ignore_index=True)
    
    # Save synthetic data
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'synthetic_data.csv')
    synthetic_data.to_csv(out_path, index=False)
    print(f"Saved synthetic data to {out_path}")
    
    # Evaluation
    print("Running evaluation...")
    target_col = args.condition_col if args.condition_col else None
    
    # Filter data for evaluation if requested
    eval_real_data = real_data
    eval_synthetic_data = synthetic_data
    
    if args.eval_filter:
        print(f"\nFiltering evaluation data by: {args.eval_filter}")
        try:
            col, val = args.eval_filter.split('=')
            col = col.strip()
            val = val.strip()
            
            # Helper to filter based on type
            def filter_df(df, col, val):
                if col not in df.columns:
                    print(f"Warning: Column {col} not found in dataframe")
                    return df
                
                # Check dtype
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = float(val)
                
                # Filter (converting to string for robust comparison if needed, or keeping type)
                return df[df[col].astype(str) == str(val)]

            eval_real_data = filter_df(real_data, col, val)
            eval_synthetic_data = filter_df(synthetic_data, col, val)
            
            print(f"Filtered Real Data: {len(eval_real_data)} samples")
            print(f"Filtered Synthetic Data: {len(eval_synthetic_data)} samples")
            
            if len(eval_real_data) < 10 or len(eval_synthetic_data) < 10:
                print("Warning: Very few samples after filtering. Metrics may be unstable.")
                
            if len(eval_synthetic_data) == 0:
                print("Error: No synthetic samples found matching the filter. Skipping evaluation.")
                return

        except ValueError:
            print("Error parsing eval_filter. Use format 'column=value'")
            return
            
    evaluator = EvaluationFramework(
        real_data=eval_real_data, 
        synthetic_data=eval_synthetic_data, 
        task_type='classification', # Assume classification for now
        target_col=target_col
    )
    
    # Fidelity
    fidelity = evaluator.evaluate_fidelity()
    print("Fidelity Metrics:", fidelity)
    
    # Utility
    utility = evaluator.evaluate_utility()
    print("Utility Metrics:", utility)
    
    # Privacy
    privacy = evaluator.evaluate_privacy(sample_size=1000)
    print("Privacy Metrics:", privacy)
    
    # Global Utility (Discriminator)
    global_utility = evaluator.evaluate_discriminator_measure()
    print("Global Utility Metrics:", global_utility)

    # Save metrics
    res = {**fidelity, **utility, **privacy, **global_utility}
    with open(os.path.join(args.output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(res, f, indent=2)
        
    # Plots
    if len(eval_synthetic_data) > 0:
        evaluator.plot_correlation_difference(os.path.join(args.output_dir, 'correlation_diff.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--preprocessor_path', type=str, required=True)
    parser.add_argument('--condition_encoder_path', type=str, default=None)
    parser.add_argument('--real_data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--condition_col', type=str, default=None)
    parser.add_argument('--sample_condition_dist', action='store_true', help='Sample conditions from real data distribution')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--eval_filter', type=str, default=None, help='Filter for evaluation (e.g., "column=value")')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    main(args)
