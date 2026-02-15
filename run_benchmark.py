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

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import EvaluationFramework
from ds_flow.config import DSFlowConfig
from ds_flow.models.dual_stream import DualStreamTransformer
from ds_flow.models.flow_matching import OTCFM
from generate_test import load_model as load_model_1, generate_synthetic_data as generate_model_1
from models.gaussian_diffusion import GaussianDiffusion

# Constants
REAL_DATA_PATH = "datasets/adult/train.csv"
BENCHMARK_DIR = "../Project/sd_framework/benchmark_results_20260202"
CONSTRAINT_FILE = os.path.join(BENCHMARK_DIR, "adult_constraints.json")
BASELINE_DATA_DIR = os.path.join(BENCHMARK_DIR, "data/adult/single")
OUTPUT_DIR = "benchmark_results"

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    engine.load_state_dict(checkpoint)
    engine.eval()
    
    return engine, tokenizer, le_condition

def generate_ds_flow_samples(engine, tokenizer, le_condition, n_samples, constraints, condition_col):
    """
    Generate samples satisfying constraints using conditional generation + rejection sampling.
    """
    device = next(engine.model.parameters()).device
    generated_df = pd.DataFrame()
    
    # Determine if we can condition
    y_cond = None
    target_class = None
    
    # Check if constraint involves condition_col
    cols = constraints['columns']
    vals = constraints['values']
    
    if condition_col in cols:
        idx = cols.index(condition_col)
        target_val = vals[idx]
        if le_condition is not None: # Ensure le_condition exists before transforming
            target_class = le_condition.transform([target_val])[0]
        
    # Generation Loop
    max_iter = 10
    
    for i in range(max_iter):
        needed = n_samples - len(generated_df)
        if needed <= 0:
            break
            
        current_batch = max(needed * 2, 100) # Heuristic
        
        # Prepare y
        y = None
        if le_condition is not None:
            if target_class is not None:
                y = torch.full((current_batch,), target_class, device=device)
            else:
                # Random y
                num_cls = len(le_condition.classes_)
                y = torch.randint(0, num_cls, (current_batch,), device=device)
            
        # Generate
        with torch.no_grad():
            tokens = engine.sample(current_batch, steps=20, y=y)
            batch_df = tokenizer.inverse_transform(tokens)
            
        # Add condition col back
        if condition_col and le_condition is not None:
            # Only if y was used/generated
             if y is not None:
                y_labels = le_condition.inverse_transform(y.cpu().numpy())
                batch_df[condition_col] = y_labels
            
        # Filter
        mask = pd.Series(True, index=batch_df.index)
        for c_name, c_val in zip(cols, vals):
            # Ensure type match (string)
            mask &= (batch_df[c_name].astype(str) == str(c_val))
            
        valid_df = batch_df[mask]
        generated_df = pd.concat([generated_df, valid_df])
        
    if len(generated_df) < n_samples:
        print(f"Warning: Only generated {len(generated_df)} valid samples for {constraints}")
        
    return generated_df.head(n_samples)

def generate_model_1_samples(model, preprocessor, diffusion, condition_encoder, n_samples, constraints, condition_col, device):
    """
    Generate samples for Model 1.
    Returns the generated DataFrame and the corresponding condition labels (if any).
    """
    generated_df = pd.DataFrame()
    generated_y_labels = []
    
    target_class = None
    cols = constraints['columns']
    vals = constraints['values']
    
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
        
        batch_df, start_y = generate_model_1(
            model, diffusion, preprocessor, current_batch, 
            guidance_scale=1.0, 
            condition_col=condition_col, 
            sample_condition_dist=(target_class is None),
            device=device,
            condition_specific_val=target_class
        )
        
        # Filter
        mask = pd.Series(True, index=batch_df.index)
        for c_name, c_val in zip(cols, vals):
            mask &= (batch_df[c_name].astype(str) == str(c_val))
            
        valid_df = batch_df[mask]
        
        # Collect y for valid samples
        if start_y is not None:
            valid_indices = mask[mask].index
            # Map original batch indices to start_y indices
            # This assumes start_y corresponds directly to batch_df rows
            # If batch_df is re-indexed, this mapping needs to be more robust
            valid_y = start_y[valid_indices.to_numpy()]
            generated_y_labels.append(valid_y)
        
        generated_df = pd.concat([generated_df, valid_df])
        
    final_df = generated_df.head(n_samples)
    final_y = None
    if generated_y_labels:
        final_y = torch.cat(generated_y_labels)[:len(final_df)]
        
    # Add condition column back
    if condition_col and final_y is not None:
         # Need condition_encoder here? Yes, passed as arg.
         labels = condition_encoder.inverse_transform(final_y.cpu().numpy())
         final_df[condition_col] = labels
        
    return final_df, final_y

def run_benchmark(args):
    # 1. Load Real Data
    print("Loading Real Data...")
    real_df = pd.read_csv(REAL_DATA_PATH)
    
    # 2. Load Constraints
    print(f"Loading Constraints from {CONSTRAINT_FILE}...")
    with open(CONSTRAINT_FILE, 'r') as f:
        constraints_data = json.load(f)
        
    constraints_list = constraints_data.get('single', []) + constraints_data.get('double', [])
    print(f"Found {len(constraints_list)} constraints.")
    
    # 3. Load Models
    print("Loading DS-Flow...")
    ds_engine, ds_tokenizer, ds_cond = load_ds_flow_model(
        args.ds_checkpoint, args.ds_tokenizer, args.ds_condition_encoder, args.d_model
    )
    
    print("Loading Model 1 (Diffusion Transformer)...")
    m1_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model1, m1_preprocessor, m1_cond_enc = load_model_1(
        args.m1_checkpoint, 
        args.m1_preprocessor, 
        args.m1_condition_encoder,
        hidden_dim=args.m1_hidden_dim,
        num_layers=args.m1_layers,
        num_heads=args.m1_heads,
        device=m1_device
    )
    m1_diffusion = GaussianDiffusion(num_timesteps=1000)
    
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
            print(f"Skipping {c_hash}: Not enough real samples.")
            continue
            
        target_n = min(len(real_subset), args.max_samples) # Cap at max_samples
        
        # --- A. DS-Flow ---
        ds_file = os.path.join(OUTPUT_DIR, f"dsflow_adult_{c_hash}.csv")
        if os.path.exists(ds_file) and not args.overwrite:
            ds_df = pd.read_csv(ds_file)
        else:
            ds_df = generate_ds_flow_samples(
                ds_engine, ds_tokenizer, ds_cond, 
                n_samples=target_n,
                constraints={'columns': cols, 'values': vals},
                condition_col=args.condition_col
            )
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            ds_df.to_csv(ds_file, index=False)
        # Metrics DS-Flow
        if len(ds_df) > 1:
            # Align columns
            ds_df = ds_df[real_subset.columns]
            
            # Subsample real_subset for speed if needed
            eval_real = real_subset
            if len(eval_real) > 2000:
                eval_real = eval_real.sample(2000, random_state=42)
            
            try:
                eval_ds = EvaluationFramework(eval_real, ds_df, task_type='classification', target_col='income')
                
                # 1. Fidelity
                fid = eval_ds.evaluate_fidelity()
                wds = [v for k, v in fid.items() if k.startswith('WD')]
                jsds = [v for k, v in fid.items() if k.startswith('JSD')]
                mean_wd = np.mean(wds) if wds else 0
                mean_jsd = np.mean(jsds) if jsds else 0
                
                # 2. Utility
                u_metrics = eval_ds.evaluate_utility()
                
                # 3. Privacy
                p_metrics = eval_ds.evaluate_privacy(sample_size=min(len(eval_real), 1000))
                
                # 4. Global Discriminator
                d_metrics = eval_ds.evaluate_discriminator_measure()
                
                results.append({
                    'Method': 'DS-Flow',
                    'Hash': c_hash,
                    'TSTR': u_metrics.get('TSTR_F1', 0),
                    'TRTR': u_metrics.get('TRTR_F1', 0),
                    'DCR_Mean': p_metrics.get('DCR_Mean', 0),
                    'DCR_5th': p_metrics.get('DCR_5th_Percentile', 0),
                    'Mean_WD': mean_wd,
                    'Mean_JSD': mean_jsd,
                    'Discrim_AUC': d_metrics.get('Discriminator_AUC', 0)
                })
            except Exception as e:
                print(f"Error evaluating DS-Flow for {c_hash}: {e}")
            
        # --- B. Model 1 ---
        m1_file = os.path.join(OUTPUT_DIR, f"model1_adult_{c_hash}.csv")
        if os.path.exists(m1_file) and not args.overwrite:
            m1_df = pd.read_csv(m1_file)
        else:
            m1_df, _ = generate_model_1_samples(
                model1, m1_preprocessor, m1_diffusion, m1_cond_enc,
                n_samples=target_n,
                constraints={'columns': cols, 'values': vals},
                condition_col=args.condition_col,
                device=m1_device
            )
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            m1_df.to_csv(m1_file, index=False)
            
        if len(m1_df) > 1:
            # Align columns
            m1_df = m1_df[real_subset.columns]
            
            # Subsample real_subset for speed if needed
            eval_real = real_subset
            if len(eval_real) > 2000:
                eval_real = eval_real.sample(2000, random_state=42)
            try:
                eval_m1 = EvaluationFramework(eval_real, m1_df, task_type='classification', target_col='income')
                
                # 1. Fidelity
                fid = eval_m1.evaluate_fidelity()
                wds = [v for k, v in fid.items() if k.startswith('WD')]
                jsds = [v for k, v in fid.items() if k.startswith('JSD')]
                mean_wd = np.mean(wds) if wds else 0
                mean_jsd = np.mean(jsds) if jsds else 0
                
                # 2. Utility
                u = eval_m1.evaluate_utility()
                
                # 3. Privacy
                p = eval_m1.evaluate_privacy(sample_size=min(len(eval_real), 1000))
                
                # 4. Global Discriminator
                d = eval_m1.evaluate_discriminator_measure()
                
                results.append({
                    'Method': 'Model1',
                    'Hash': c_hash,
                    'TSTR': u.get('TSTR_F1', 0),
                    'TRTR': u.get('TRTR_F1', 0),
                    'DCR_Mean': p.get('DCR_Mean', 0),
                    'DCR_5th': p.get('DCR_5th_Percentile', 0),
                    'Mean_WD': mean_wd,
                    'Mean_JSD': mean_jsd,
                    'Discrim_AUC': d.get('Discriminator_AUC', 0)
                })
            except Exception as e:
                print(f"Error evaluating Model 1 for {c_hash}: {e}")
            
    # Save Results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "benchmark_comparison.csv"), index=False)
    print("Benchmark Complete!")
    try:
        print(res_df.groupby('Method').mean(numeric_only=True))
    except Exception as e:
        print(f"Could not print summary: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DS-Flow Args
    parser.add_argument('--ds_checkpoint', type=str, required=True)
    parser.add_argument('--ds_tokenizer', type=str, required=True)
    parser.add_argument('--ds_condition_encoder', type=str, default=None)
    parser.add_argument('--d_model', type=int, default=512)
    # Model 1 Args
    parser.add_argument('--m1_checkpoint', type=str, required=True)
    parser.add_argument('--m1_preprocessor', type=str, required=True)
    parser.add_argument('--m1_condition_encoder', type=str, required=True)
    parser.add_argument('--m1_hidden_dim', type=int, default=512)
    parser.add_argument('--m1_layers', type=int, default=6)
    parser.add_argument('--m1_heads', type=int, default=8)
    # General
    parser.add_argument('--condition_col', type=str, default='income')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max_samples', type=int, default=1000)
    
    args = parser.parse_args()
    run_benchmark(args)


