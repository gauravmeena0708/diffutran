import torch
import pandas as pd
import argparse
import os
import sys
import joblib
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ds_flow.config import DSFlowConfig
from ds_flow.models.dual_stream import DualStreamTransformer
from ds_flow.models.flow_matching import OTCFM
from ds_flow.data.tokenizer import QuantileTokenizer

def sample_ds_flow(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # 3. Model
    print("Initializing Model...")
    backbone = DualStreamTransformer(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout
    ).to(device)
    
    engine = OTCFM(backbone, tokenizer, config).to(device)
    
    # 4. Load Weights
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    engine.load_state_dict(checkpoint)
    
    # Condition Handling
    y_cond = None
    le_condition = None
    condition_col = args.condition_col
    
    if condition_col:
        dir_name = os.path.dirname(args.checkpoint_path)
        encoder_path = os.path.join(dir_name, 'condition_encoder.joblib')
        # Try both .json and .joblib for compatibility
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
            
            # Generate y
            # Simple approach: Sample uniformly (or from specific distribution if implemented)
            # For now: Uniform
            print(f"Sampling {condition_col} uniformly from {num_classes} classes...")
            y_cond = torch.randint(0, num_classes, (args.num_samples,), device=device)
        else:
            print(f"Warning: condition_col provided but {encoder_path} not found.")

    # 5. Sample
    print(f"Generating {args.num_samples} samples...")
    x_tokens = engine.sample(args.num_samples, steps=args.steps, y=y_cond)
    
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

    output_path = os.path.join(args.output_dir, args.output_name)
    df_syn.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_dsflow')
    parser.add_argument('--output_name', type=str, default='synthetic_data.csv')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--d_model', type=int, default=512)
    
    args = parser.parse_args()
    sample_ds_flow(args)
