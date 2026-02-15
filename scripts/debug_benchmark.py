
import os
import sys
import torch
import pandas as pd
import json
import joblib

# Add project root
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.flow_model.models.dual_stream import DualStreamTransformer
from src.flow_model.models.flow_matching import OTCFM
from src.flow_model.data.tokenizer import QuantileTokenizer
from src.flow_model.config import DSFlowConfig

def load_ds_flow_model(checkpoint_path, tokenizer_path, condition_encoder_path=None, d_model=512):
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    tokenizer = QuantileTokenizer.load(tokenizer_path, device=device)
    print("Tokenizer loaded.")
    
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
    print("Condition Encoder loaded.")

    config = DSFlowConfig()
    config.d_model = d_model
    
    backbone = DualStreamTransformer(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        num_classes=num_classes
    ).to(device)
    
    engine = OTCFM(backbone, tokenizer, config).to(device)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    engine.load_state_dict(checkpoint)
    engine.eval()
    print("Model loaded successfully.")
    
    return engine, tokenizer, le_condition

if __name__ == "__main__":
    try:
        engine, tokenizer, le_condition = load_ds_flow_model(
            "outputs/checkpoints_dsflow_full/model_final.pt",
            "outputs/checkpoints_dsflow_full/tokenizer.pt",
            "outputs/checkpoints_dsflow_full/condition_encoder.joblib"
        )
        
        print("Attempting to sample 10 items...")
        device = next(engine.model.parameters()).device
        with torch.no_grad():
            tokens = engine.sample(10, steps=20)
            df = tokenizer.inverse_transform(tokens)
        print("Sampling successful!")
        print(df.head())
        
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()
