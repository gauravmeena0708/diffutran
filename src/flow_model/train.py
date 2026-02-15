import os
import sys
import torch
import torch.optim as optim
import argparse
import pandas as pd
from tqdm import tqdm
import json
from sklearn.preprocessing import LabelEncoder

from .config import DSFlowConfig
from .data.tokenizer import QuantileTokenizer
from .data.loader import get_dataloader
from .models.dual_stream import DualStreamTransformer
from .models.flow_matching import OTCFM

def train_ds_flow(args):
    # 1. Config
    config = DSFlowConfig()
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.d_model = args.d_model
    config.n_layers = args.n_layers
    config.n_heads = args.n_heads
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Condition Column
    condition_col = args.condition_col
    le_condition = None
    num_classes = None
    
    if condition_col:
        print(f"Conditioning on: {condition_col}")
        le_condition = LabelEncoder()
        # Ensure string
        df[condition_col] = df[condition_col].astype(str)
        le_condition.fit(df[condition_col])
        num_classes = len(le_condition.classes_)
        
        # Save condition encoder
        with open(os.path.join(args.save_dir, 'condition_encoder.json'), 'w') as f:
            json.dump(le_condition.classes_.tolist(), f)
        
    # Initialize Tokenizer (Fit on data WITHOUT condition col? Or with?)
    # If we want to condition, we should NOT tokenize the condition column as a feature.
    # So we should exclude it from categorical_cols/numerical_cols in tokenizer fit.
    
    
    # Filter columns for tokenizer
    cat_cols = args.categorical_cols if args.categorical_cols else []
    
    if condition_col:
        # If passed as string in sh, it might be a single string or list?
        # argparse nargs='+' returns a list.
        if condition_col in cat_cols:
            cat_cols.remove(condition_col)
        # If it was numerical, we treat it as categorical for conditioning.
            
    tokenizer = QuantileTokenizer(
        d_model=config.d_model, 
        device=device
    )
    
    print("Fitting tokenizer...")
    # Pass df but specify cols
    # Tokenizer fits on specified cols only?
    # QuantileTokenizer logic:
    # "if categorical_cols is None: ... numerical_cols = [c for c in df.columns if c not in categorical_cols]"
    # So if we pass categorical_cols, it infers numerical from the rest.
    # We must drop condition_col from DF passed to fit, OR manually set numerical_cols in tokenizer?
    # Tokenizer fit implementation:
    # "self.numerical_cols = [c for c in df.columns if c not in categorical_cols]"
    # So we MUST drop condition_col from df before passing to fit.
    
    df_features = df.drop(columns=[condition_col]) if condition_col else df
    tokenizer.fit(df_features, categorical_cols=cat_cols)
    
    # Save Tokenizer
    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.save_dir, 'tokenizer.pt')
    tokenizer.save(tokenizer_path)
    
    # Create DataLoader
    # We pass the FULL df to loader, but loader needs to know what to do.
    # DSDataset: "tokenizer.preprocess(df)". 
    # Tokenizer preprocesses columns present in self.numerical_cols/cat_cols.
    # Since we excluded condition_col from tokenizer, preprocess will ignore it (if using column names).
    # Wait, Tokenizer stores column names.
    # Preprocess logic: "for col in self.numerical_cols: ... df[col]..."
    # So passing full df to dataset is fine if tokenizer ignores extra columns.
    # Let's check DSDataset init: "tokenizer.preprocess(df)".
    # Tokenizer uses stored column names.
    # So we can pass full df.
    
    print("Creating DataLoader...")
    dataloader = get_dataloader(
        df, 
        tokenizer, 
        batch_size=config.batch_size, 
        condition_col=condition_col, 
        label_encoder=le_condition
    )
    
    # 3. Model
    print("Initializing Dual-Stream Transformer...")
    backbone = DualStreamTransformer(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        num_classes=num_classes
    ).to(device)
    
    # Engine
    engine = OTCFM(backbone, tokenizer, config).to(device)
    
    # 4. Optimizer
    optimizer = optim.AdamW(
        engine.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloader) * config.num_epochs
    )
    
    # 5. Training Loop
    print("Starting Training...")
    history = []
    
    for epoch in range(config.num_epochs):
        engine.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in pbar:
            x_num, x_cat, y = batch
            
            # Loss Calculation
            loss, logs = engine.compute_loss(x_num, x_cat, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(logs)
            
        avg_loss = epoch_loss / len(dataloader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            path = os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save(engine.state_dict(), path)
            print(f"Saved checkpoint to {path}")
            
    # Save Final
    torch.save(engine.state_dict(), os.path.join(args.save_dir, 'model_final.pt'))
    print("Training Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints_dsflow')
    parser.add_argument('--categorical_cols', type=str, nargs='+', default=None)
    parser.add_argument('--condition_col', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    
    args = parser.parse_args()
    train_ds_flow(args)
