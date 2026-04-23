import torch
import pandas as pd
import argparse
import os
import sys
import joblib
import json
import numpy as np

# Add project root to path to allow imports from scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.experiment_utils import get_constraint_hash
from .config import DSFlowConfig
from .models.dual_stream import DualStreamTransformer
from .models.flow_matching import OTCFM
from .data.tokenizer import QuantileTokenizer

def sample_ds_flow(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    condition_col = args.condition_col
    
    # 1. Load Tokenizer (Sklearn state + Structure)
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    # Warning: joblib loading a nn.Module might require the class definition to be exactly same
    # and might have issues if saved on GPU and loaded on CPU or vice versa.
    # We assume it was saved on CPU/GPU and we handle mapping.
    
    # Use the safe load method
    tokenizer = QuantileTokenizer.load(args.tokenizer_path, device=device)
    
    # 2. Config
    config = DSFlowConfig()
    config.d_model = args.d_model # Ensure this matches training!
    # Ideally config should be saved too.
    
    # Determine num_classes from encoder if conditioning
    num_classes = None
    le_condition = None
    if condition_col:
        dir_name = os.path.dirname(args.checkpoint_path)
        encoder_path = os.path.join(dir_name, 'condition_encoder.joblib')
        if not os.path.exists(encoder_path):
            encoder_path = os.path.join(dir_name, 'condition_encoder.json')

        if os.path.exists(encoder_path):
            print(f"Loading condition encoder from {encoder_path}")
            if encoder_path.endswith('.json'):
                from sklearn.preprocessing import LabelEncoder
                le_condition = LabelEncoder()
                with open(encoder_path, 'r') as f:
                    classes = json.load(f)
                le_condition.classes_ = np.array(classes)
            else:
                le_condition = joblib.load(encoder_path)
            num_classes = len(le_condition.classes_)

    # 3. Model
    print("Initializing Model...")
    backbone = DualStreamTransformer(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        num_classes=num_classes
    ).to(device)
    
    engine = OTCFM(backbone, tokenizer, config).to(device)
    
    # 4. Load Weights
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    engine.load_state_dict(checkpoint)
    
    # Condition Handling
    y_cond = None
    if condition_col and num_classes is not None:
        # Generate y
        print(f"Sampling {condition_col} uniformly from {num_classes} classes...")
        y_cond = torch.randint(0, num_classes, (args.num_samples,), device=device)

    # Constraint Handling
    cat_constraints = {}
    num_constraints = {}
    if args.constraint:
        for c in args.constraint:
            if '=' in c:
                col, val = c.split('=', 1)
                col, val = col.strip(), val.strip()
                if col in tokenizer.categorical_cols:
                    cat_constraints[col] = val
                elif col in tokenizer.numerical_cols:
                    num_constraints[col] = float(val)

    # 5. Sample
    print(f"Generating {args.num_samples} samples using {args.solver} solver with {args.steps} steps...")
    x_tokens = engine.sample(
        args.num_samples, 
        steps=args.steps, 
        y=y_cond, 
        solver=args.solver,
        cat_constraints=cat_constraints if cat_constraints else None,
        num_constraints=num_constraints if num_constraints else None
    )
    
    # 6. Reconstruct
    print("Reconstructing DataFrame...")
    df_syn = tokenizer.inverse_transform(x_tokens)
    
    # Add condition column back
    if condition_col and y_cond is not None:
        y_np = y_cond.cpu().numpy()
        labels = le_condition.inverse_transform(y_np)
        df_syn[condition_col] = labels
    
    # 7. Save
    os.makedirs(args.output_dir, exist_ok=True)

    # New naming convention based on sd_framework
    from datetime import datetime

    model_name = "flow_model"
    dataset_name = args.dataset_name
    
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

    output_path = os.path.join(args.output_dir, output_filename)
    df_syn.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset, used for naming the output file.")
    parser.add_argument('--output_dir', type=str, default='results_dsflow')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--solver', type=str, default='dpm++', choices=['euler', 'dpm++'], help="ODE solver to use for continuous numerical variables.")
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--condition_col', type=str, default=None)
    parser.add_argument('--constraint', type=str, action='append', help='Add a constraint in format column=value. Can be used multiple times.')
    
    args = parser.parse_args()
    sample_ds_flow(args)
