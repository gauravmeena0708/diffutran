import torch
import numpy as np
import pandas as pd
import os
import argparse
import json
import matplotlib.pyplot as plt
import sys

# Make the project root directory discoverable by Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Modular imports from the new 'src' directory
from src.diffutron.model import DiffusionModel
from src.diffutron.diffusion import GaussianDiffusion
from src.common.data_utils import DataPreprocessor
from src.common.evaluation_metrics import EvaluationFramework
from scripts.experiment_utils import get_constraint_hash


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
        from sklearn.preprocessing import LabelEncoder
        condition_encoder = LabelEncoder()

        # Safe loading from JSON
        if condition_encoder_path.endswith('.json'):
            with open(condition_encoder_path, 'r') as f:
                classes = json.load(f)
            condition_encoder.classes_ = np.array(classes)
        else:
            # Fallback for old format if necessary, though new format is preferred
            import joblib
            condition_encoder = joblib.load(condition_encoder_path)

        num_classes = len(condition_encoder.classes_)
    
    # Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
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

def generate_synthetic_data(model, diffusion, preprocessor, num_samples, guidance_scale=1.0, condition_col=None, condition_encoder=None, sample_condition_dist=False, device='cuda', condition_specific_val=None):
    model.eval()
    
    # 1. Generate Conditions (y)
    y = None
    if condition_col:
        # If specific value requested (for benchmarking/subgroups)
        if condition_specific_val is not None:
            # Assume it's already an integer index
            y = torch.full((num_samples,), condition_specific_val, device=device).long()
        elif sample_condition_dist:
             # Sample from distribution of real data? 
             # For now, uniform if not provided
             y = torch.randint(0, model.num_classes, (num_samples,), device=device).long()
        else:
             # Balanced?
             y = torch.randint(0, model.num_classes, (num_samples,), device=device).long()
             
    # 2. Sample Noise
    # Infer dims from preprocessor
    num_num = len(preprocessor.numerical_cols)
    cards = preprocessor.categorical_cardinalities
    num_cat = sum(cards.values()) if isinstance(cards, dict) else sum(cards)
    x = torch.randn((num_samples, num_num + num_cat), device=device)
    
    # 3. Denoise loop
    for i in reversed(range(diffusion.num_timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        x = diffusion.p_sample(model, x, t, y, guidance_scale=guidance_scale)
        
    # 4. Inverse Transform
    df_syn = preprocessor.inverse_transform(x.cpu().numpy())
    
    # 5. Add Condition Column
    if condition_col and y is not None and condition_encoder is not None:
        y_np = y.cpu().numpy()
        df_syn[condition_col] = condition_encoder.inverse_transform(y_np)
        
    return df_syn, y

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
    
    # Load real data for evaluation
    print(f"Loading real data from {args.real_data_path}...")
    real_data = pd.read_csv(args.real_data_path)

    synthetics = []
    
    # Generate in batches
    batch_size = args.batch_size
    num_generated = 0
    
    while num_generated < args.num_samples:
        current_batch_size = min(batch_size, args.num_samples - num_generated)
        
        # Prepare batch conditions if needed
        # Note: generate_synthetic_data handles condition generation internally mostly.
        # But if we want to pass specific conditions like in original main, we need to adapt.
        # Original main had:
        # if args.sample_condition_dist: ... conditions = ...
        
        # We need to bridge the gap.
        # generate_synthetic_data doesn't accept "condition" array directly in my signature above.
        # It accepts condition_specific_val or sample_condition_dist flag.
        
        # For this script (generate_test.py) usage, we might need to pass `y` explicitly?
        # Let's keep it simple and just call generate_synthetic_data with flags.
        
        # syn_batch is now (df, y_tensor)
        syn_batch_df, batch_y = generate_synthetic_data(
            model, diffusion, preprocessor, 
            num_samples=current_batch_size, 
            guidance_scale=args.guidance_scale,
            condition_col=args.condition_col,
            condition_encoder=condition_encoder,
            sample_condition_dist=args.sample_condition_dist,
            device=device
        )
            
        synthetics.append(syn_batch_df)
        num_generated += current_batch_size
        
    synthetic_data = pd.concat(synthetics, ignore_index=True)

    
    # Save synthetic data
    os.makedirs(args.output_dir, exist_ok=True)
    
    # New naming convention based on sd_framework
    from datetime import datetime

    def get_dataset_name(path):
        return os.path.basename(os.path.dirname(path))

    model_name = "diffutron"
    dataset_name = get_dataset_name(args.real_data_path)
    
    # Process constraints for hashing
    constraints_for_hash = []
    if args.constraint:
        for c in args.constraint:
            if '=' in c:
                col, val = c.split('=', 1)
                constraints_for_hash.append({"column": col.strip(), "value": val.strip()})

    cons_hash = get_constraint_hash(constraints_for_hash) if constraints_for_hash else "baseline"
    
    month = datetime.now().strftime("%Y-%m")
    num_samples = args.num_samples
    
    output_filename = f"{model_name}_{dataset_name}_{cons_hash}_{month}_{num_samples}.csv"
    
    out_path = os.path.join(args.output_dir, output_filename)
    synthetic_data.to_csv(out_path, index=False)
    print(f"Saved synthetic data to {out_path}")
    
    # Evaluation
    print("Running evaluation...")
    target_col = args.condition_col if args.condition_col else None

    # Load real data
    try:
        real_data = pd.read_csv(args.real_data_path)
    except Exception as e:
        print(f"Error loading real data from {args.real_data_path}: {e}")
        return

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

    # Relationship Report
    print("Generating Relationship Report...")
    report = evaluator.generate_relationship_report(data_type='synthetic')
    with open(os.path.join(args.output_dir, 'relationship_report.txt'), 'w') as f:
        f.write(report)
    print(f"Saved relationship report to {os.path.join(args.output_dir, 'relationship_report.txt')}")

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
    parser.add_argument('--constraint', type=str, action='append', help='Add a constraint in format column=value. Can be used multiple times.')
    parser.add_argument('--sample_condition_dist', action='store_true', help='Sample conditions from real data distribution')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--eval_filter', type=str, default=None, help='Filter for evaluation (e.g., "column=value")')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    main(args)
