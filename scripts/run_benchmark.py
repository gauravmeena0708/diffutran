import argparse
import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm

# Add project root to path
# This allows importing from 'src'
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.common.evaluation_metrics import EvaluationFramework
from src.flow_model.config import DSFlowConfig
from src.flow_model.models.dual_stream import DualStreamTransformer
from src.flow_model.models.flow_matching import OTCFM
from src.flow_model.data.tokenizer import QuantileTokenizer
from src.diffutron.diffusion import GaussianDiffusion
from generate_test import load_model as load_baseline_model, generate_synthetic_data as generate_baseline_data

def load_ds_flow_model(checkpoint_path, tokenizer_path, condition_encoder_path=None, d_model=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Tokenizer
    tokenizer = QuantileTokenizer.load(tokenizer_path, device=device)
    
    # Load Condition Encoder
    le_condition = None
    num_classes = None
    if condition_encoder_path and os.path.exists(condition_encoder_path):
        from sklearn.preprocessing import LabelEncoder
        le_condition = LabelEncoder()
        if condition_encoder_path.endswith('.json'):
            with open(condition_encoder_path, 'r') as f:
                classes = json.load(f)
            le_condition.classes_ = np.array(classes)
        else:
            le_condition = joblib.load(condition_encoder_path)
        num_classes = len(le_condition.classes_)
    
    # Config
    config = DSFlowConfig()
    config.d_model = d_model
    
    # Model
    backbone = DualStreamTransformer(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        num_classes=num_classes
    ).to(device)
    
    engine = OTCFM(backbone, tokenizer, config).to(device)
    
    # Weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    engine.load_state_dict(checkpoint)
    engine.eval()
    
    return engine, tokenizer, le_condition

def generate_ds_flow_samples(engine, tokenizer, le_condition, n_samples, constraints, condition_col):
    """
    Generate samples satisfying constraints using conditional generation + rejection sampling.
    """
    device = next(engine.model.parameters()).device
    generated_df = pd.DataFrame()
    
    # Check if constraint involves condition_col
    cols = constraints['columns']
    vals = constraints['values']
    
    target_class = None
    if condition_col in cols:
        idx = cols.index(condition_col)
        target_val = vals[idx]
        if le_condition is not None:
            # Transform target value to class index
            try:
                target_class = le_condition.transform([target_val])[0]
            except ValueError:
                 print(f"Warning: Value {target_val} not in condition encoder classes.")

    # Generation Loop
    max_iter = 10
    
    for i in range(max_iter):
        needed = n_samples - len(generated_df)
        if needed <= 0:
            break
            
        current_batch = max(needed * 2, 100)
        
        # Prepare condition y
        y = None
        if le_condition is not None:
            if target_class is not None:
                y = torch.full((current_batch,), target_class, device=device)
            else:
                # Random y if not constrained on condition_col
                num_cls = len(le_condition.classes_)
                y = torch.randint(0, num_cls, (current_batch,), device=device)
            
        # Generate
        with torch.no_grad():
            tokens = engine.sample(current_batch, steps=20, y=y)
            batch_df = tokenizer.inverse_transform(tokens)
            
        # Add condition col back if it was handled separately
        if condition_col and le_condition is not None and y is not None:
             y_labels = le_condition.inverse_transform(y.cpu().numpy())
             batch_df[condition_col] = y_labels
            
        # Filter by constraints
        mask = pd.Series(True, index=batch_df.index)
        for c_name, c_val in zip(cols, vals):
            mask &= (batch_df[c_name].astype(str) == str(c_val))
            
        valid_df = batch_df[mask]
        generated_df = pd.concat([generated_df, valid_df])
        
    if len(generated_df) < n_samples:
        print(f"Warning: Only generated {len(generated_df)} valid samples for {constraints}")
        
    return generated_df.head(n_samples)

def generate_baseline_samples(model, preprocessor, diffusion, condition_encoder, n_samples, constraints, condition_col, device):
    """
    Generate samples for the Baseline Diffusion Model.
    """
    generated_df = pd.DataFrame()
    
    cols = constraints['columns']
    vals = constraints['values']
    
    target_class = None
    if condition_col and condition_col in cols:
        idx = cols.index(condition_col)
        target_val = vals[idx]
        if str(target_val) in condition_encoder.classes_:
            target_class = condition_encoder.transform([str(target_val)])[0]
            
    max_iter = 10
    for i in range(max_iter):
        needed = n_samples - len(generated_df)
        if needed <= 0:
            break
            
        current_batch = max(needed * 2, 100)
        
        batch_df, _ = generate_baseline_data(
            model, diffusion, preprocessor, current_batch, 
            guidance_scale=1.0, 
            condition_col=condition_col, 
            condition_encoder=condition_encoder,
            sample_condition_dist=(target_class is None),
            device=device,
            condition_specific_val=target_class
        )
        
        # Filter
        mask = pd.Series(True, index=batch_df.index)
        for c_name, c_val in zip(cols, vals):
            mask &= (batch_df[c_name].astype(str) == str(c_val))
            
        valid_df = batch_df[mask]
        generated_df = pd.concat([generated_df, valid_df])
        
    return generated_df.head(n_samples)

def evaluate_generation(real_subset, synthetic_df, method_name, constraint_hash):
    """
    Compute Fidelity, Utility, and Privacy metrics for a generated dataset.
    Returns a dictionary of results.
    """
    if len(synthetic_df) < 2:
        return None
        
    # Align columns
    try:
        common_cols = [c for c in real_subset.columns if c in synthetic_df.columns]
        if len(common_cols) < len(real_subset.columns):
            print(f"[{method_name}] Missing columns. Skipping.")
            return None
            
        synthetic_df = synthetic_df[real_subset.columns]
        
        # Subsample real_subset for speed if needed
        eval_real = real_subset
        if len(eval_real) > 2000:
            eval_real = eval_real.sample(2000, random_state=42)
            
        evaluator = EvaluationFramework(eval_real, synthetic_df, task_type='classification', target_col='income')
        
        # 1. Fidelity
        fid = evaluator.evaluate_fidelity()
        wds = [v for k, v in fid.items() if k.startswith('WD')]
        jsds = [v for k, v in fid.items() if k.startswith('JSD')]
        mean_wd = np.mean(wds) if wds else 0
        mean_jsd = np.mean(jsds) if jsds else 0
        
        # 2. Utility
        u_metrics = evaluator.evaluate_utility()
        
        # 3. Privacy
        p_metrics = evaluator.evaluate_privacy(sample_size=min(len(eval_real), 1000))
        
        # 4. Global Discriminator
        d_metrics = evaluator.evaluate_discriminator_measure()
        
        return {
            'Method': method_name,
            'Hash': constraint_hash,
            'TSTR': u_metrics.get('TSTR_F1', 0),
            'TRTR': u_metrics.get('TRTR_F1', 0),
            'DCR_Mean': p_metrics.get('DCR_Mean', 0),
            'DCR_5th': p_metrics.get('DCR_5th_Percentile', 0),
            'Mean_WD': mean_wd,
            'Mean_JSD': mean_jsd,
            'Discrim_AUC': d_metrics.get('Discriminator_AUC', 0)
        }
    except Exception as e:
        print(f"Error evaluating {method_name} for {constraint_hash}: {e}")
        return None

def run_benchmark(args):
    # 1. Load Real Data
    print(f"Loading Real Data from {args.real_data_path}...")
    real_df = pd.read_csv(args.real_data_path)
    
    # 2. Load Constraints
    constraint_file = os.path.join(args.benchmark_dir, args.constraints_file)
    print(f"Loading Constraints from {constraint_file}...")
    with open(constraint_file, 'r') as f:
        constraints_data = json.load(f)
        
    constraints_list = constraints_data.get('single', []) + constraints_data.get('double', [])
    print(f"Found {len(constraints_list)} constraints.")
    
    if args.target_hash:
        constraints_list = [c for c in constraints_list if c.get('hash') == args.target_hash]
        print(f"Filtered for constraint hash: {args.target_hash}")
        if not constraints_list:
            print("Warning: Target hash not found!")
            return
            
    if args.shuffle and not args.target_hash:
        import random
        random.seed(42)
        random.shuffle(constraints_list)
        print("Shuffled constraints.")
    
    # 3. Load Models
    print("Loading DS-Flow...")
    ds_engine, ds_tokenizer, ds_cond = load_ds_flow_model(
        args.ds_checkpoint, args.ds_tokenizer, args.ds_condition_encoder, args.d_model
    )
    
    print("Loading Baseline Diffusion (Model 1)...")
    base_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_model, base_preprocessor, base_cond_enc = load_baseline_model(
        args.baseline_checkpoint, 
        args.baseline_preprocessor, 
        args.baseline_condition_encoder,
        hidden_dim=args.baseline_hidden_dim,
        num_layers=args.baseline_layers,
        num_heads=args.baseline_heads,
        device=base_device
    )
    base_diffusion = GaussianDiffusion(num_timesteps=1000)
    
    results = []
    if args.limit:
        constraints_list = constraints_list[:args.limit]
        
    for c_idx, constraint in tqdm(enumerate(constraints_list), total=len(constraints_list)):
        c_hash = constraint['hash']
        cols = constraint['columns']
        vals = constraint['values']
        
        # Target Real Data
        real_subset = real_df.copy()
        for c, v in zip(cols, vals):
            real_subset = real_subset[real_subset[c].astype(str) == str(v)]
            
        if len(real_subset) < 10:
            print(f"Skipping {c_hash}: Not enough real samples ({len(real_subset)}).")
            continue
            
        target_n = min(len(real_subset), args.max_samples)
        
        # --- A. Evaluation Loop ---
        files_to_eval = [] # List of (method_name, dataframe)
        
        # 1. DS-Flow
        ds_file = os.path.join(args.output_dir, f"dsflow_adult_{c_hash}.csv")
        if os.path.exists(ds_file) and not args.overwrite:
            ds_df = pd.read_csv(ds_file)
        else:
            ds_df = generate_ds_flow_samples(
                ds_engine, ds_tokenizer, ds_cond, 
                n_samples=target_n,
                constraints={'columns': cols, 'values': vals},
                condition_col=args.condition_col
            )
            os.makedirs(args.output_dir, exist_ok=True)
            ds_df.to_csv(ds_file, index=False)
            
        files_to_eval.append(('DS-Flow', ds_df))
        
        # 2. Baseline Diffusion
        base_file = os.path.join(args.output_dir, f"baseline_adult_{c_hash}.csv")
        if os.path.exists(base_file) and not args.overwrite:
            base_df = pd.read_csv(base_file)
        else:
            base_df = generate_baseline_samples(
                base_model, base_preprocessor, base_diffusion, base_cond_enc,
                n_samples=target_n,
                constraints={'columns': cols, 'values': vals},
                condition_col=args.condition_col,
                device=base_device
            )
            os.makedirs(args.output_dir, exist_ok=True)
            base_df.to_csv(base_file, index=False)
            
        files_to_eval.append(('Baseline_Diffusion', base_df))
        
        # 3. External Models (Dynamic Discovery)
        # Search relative to benchmark dir
        baseline_data_dir = os.path.join(args.benchmark_dir, args.baseline_data_dir)
        search_dirs = [
             baseline_data_dir,
            baseline_data_dir.replace('/single', '/double') # Naive support for double constraints folder structure
        ]
        
        for s_dir in search_dirs:
            if not os.path.exists(s_dir):
                continue
            
            pattern = os.path.join(s_dir, f"*_adult_{c_hash}_*.csv")
            matches = glob.glob(pattern)
            
            for m_file in matches:
                basename = os.path.basename(m_file)
                # Parse model name
                if '_adult_' in basename:
                    model_name = basename.split('_adult_')[0]
                else:
                    model_name = "External"
                
                try:
                    ext_df = pd.read_csv(m_file)
                    # Normalize columns if needed (dots to hyphens)
                    ext_df.columns = [c.replace('.', '-') for c in ext_df.columns]
                    files_to_eval.append((model_name, ext_df))
                except Exception as e:
                    print(f"Failed to read external file {basename}: {e}")

        # --- Run Evaluation ---
        for method, df in files_to_eval:
            res = evaluate_generation(real_subset, df, method, c_hash)
            if res:
                results.append(res)
            
    # Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    res_df = pd.DataFrame(results)
    out_file = os.path.join(args.output_dir, "benchmark_comparison.csv")
    res_df.to_csv(out_file, index=False)
    print(f"Benchmark Complete! Results saved to {out_file}")
    
    try:
        if not res_df.empty:
            print("\n--- Summary ---")
            print(res_df.groupby('Method').mean(numeric_only=True))
    except Exception as e:
        print(f"Could not print summary: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Paths (Parametrized)
    parser.add_argument('--real_data_path', type=str, default="datasets/adult/train.csv")
    parser.add_argument('--benchmark_dir', type=str, default="../Project/sd_framework/benchmark_results_20260202")
    parser.add_argument('--constraints_file', type=str, default="adult_constraints.json")
    parser.add_argument('--baseline_data_dir', type=str, default="data/adult/single")
    parser.add_argument('--output_dir', type=str, default="benchmark_results")
    
    # DS-Flow Args
    parser.add_argument('--ds_checkpoint', type=str, required=True)
    parser.add_argument('--ds_tokenizer', type=str, required=True)
    parser.add_argument('--ds_condition_encoder', type=str, default=None)
    parser.add_argument('--d_model', type=int, default=512)
    
    # Baseline Model Args (Renamed from m1_)
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--baseline_preprocessor', type=str, required=True)
    parser.add_argument('--baseline_condition_encoder', type=str, required=True)
    parser.add_argument('--baseline_hidden_dim', type=int, default=512)
    parser.add_argument('--baseline_layers', type=int, default=6)
    parser.add_argument('--baseline_heads', type=int, default=8)
    
    # General
    parser.add_argument('--condition_col', type=str, default='income')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle constraints before limiting')
    parser.add_argument('--target_hash', type=str, default=None, help='Run specific constraint hash')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max_samples', type=int, default=1000)
    
    args = parser.parse_args()
    run_benchmark(args)
