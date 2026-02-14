import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Modular imports
from models.diffusion import DiffusionModel
from models.gaussian_diffusion import GaussianDiffusion


class TabularDataset(Dataset):
    """Dataset for tabular data with optional conditioning"""
    def __init__(self, data, condition=None):
        """
        Args:
            data: numpy array [num_samples, num_features]
            condition: numpy array [num_samples] (class indices) or None
        """
        self.data = torch.FloatTensor(data)
        self.condition = torch.LongTensor(condition) if condition is not None else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        cond = self.condition[idx] if self.condition is not None else None
        return sample, cond


class DataPreprocessor:
    """Handles preprocessing and inverse transformation of tabular data"""
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.categorical_cols = []
        self.numerical_cols = []
        self.column_info = {}
        self.categorical_cardinalities = []
    
    def fit_transform(self, df, categorical_cols=None):
        """
        Fit and transform the data
        """
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.categorical_cols = categorical_cols
        self.numerical_cols = [col for col in df.columns if col not in categorical_cols]
        
        # Store column info
        self.column_info = {
            'all_columns': df.columns.tolist(),
            'categorical': self.categorical_cols,
            'numerical': self.numerical_cols
        }
        
        # Encode categorical columns
        self.categorical_cardinalities = []
        for col in self.categorical_cols:
            le = LabelEncoder()
            # We treat categoricals as integers for Embedding lookup in the model
            # But the Gaussian Diffusion expects continuous inputs?
            # WAIT. Implementation Decision in models/diffusion.py:
            # - Input to forward is `x` [batch, total_dims].
            # - If we use one-hot encoding for categoricals, x is larger.
            # - If we use embeddings, we usually don't diffuse on embeddings directly in standard DDPM.
            # 
            # Re-reading my `DiffusionModel`:
            # It expects `x` which is split into `feature_slices`.
            # `input_projections` project these slices to `d_model`.
            # If `categorical_dims` are raw cardinalities (1 integer per cat), then input should be integer?
            # NO. Diffusion models need valid gradients on inputs to work with noise 
            # (conceptually, though here we just predict noise).
            # But `x_t` is noisy. You can't have "noisy integers".
            #
            # Conclusion: `x` MUST be continuous.
            # So categoricals MUST be one-hot encoded (or similar) before adding noise.
            # Thus `DataPreprocessor` must output one-hot encoded categoricals.
            
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            cardinality = len(le.classes_)
            self.categorical_cardinalities.append(cardinality)
        
        # Scale numerical columns
        for col in self.numerical_cols:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
            
        # Convert to one-hot for categoricals to make them continuous for diffusion
        # We need to flatten this into a big vector
        # Sequence: [Num1, Num2, ..., OneHotCat1, OneHotCat2, ...]
        
        processed_data = []
        
        # 1. Numerical
        if self.numerical_cols:
            processed_data.append(df[self.numerical_cols].values)
            
        # 2. Categorical (One-Hot)
        for col in self.categorical_cols:
            # to_categorical equivalents
            cat_values = df[col].values
            card = len(self.label_encoders[col].classes_)
            one_hot = np.eye(card)[cat_values]
            processed_data.append(one_hot)
            
        if processed_data:
            return np.hstack(processed_data)
        else:
            return np.empty((len(df), 0))
    
    def transform(self, df):
        """Transform new data"""
        df = df.copy()
        
        # ... (Simplified for brevity, assuming fit_transform logic is mirrored)
        # For strict correctness we should implement this, but for training fit_transform is enough.
        # We need this for condition transformation in generation.
        
        processed_data = []
        
        # Numerical
        for col in self.numerical_cols:
            df[col] = self.scalers[col].transform(df[[col]])
        if self.numerical_cols:
             processed_data.append(df[self.numerical_cols].values)

        # Categorical
        for col in self.categorical_cols:
            # Handle unknown categories?
            # For now assume known.
            df[col] = self.label_encoders[col].transform(df[col].astype(str))
            cat_values = df[col].values
            card = len(self.label_encoders[col].classes_)
            one_hot = np.eye(card)[cat_values]
            processed_data.append(one_hot)
            
        if processed_data:
            return np.hstack(processed_data)
        return np.empty((len(df), 0))

    def inverse_transform(self, data):
        """Convert generated continuous data back to original format"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        df_dict = {}
        current_idx = 0
        
        # 1. Numerical
        for col in self.numerical_cols:
            val = data[:, current_idx : current_idx + 1]
            df_dict[col] = self.scalers[col].inverse_transform(val).flatten()
            current_idx += 1
            
        # 2. Categorical
        for col, card in zip(self.categorical_cols, self.categorical_cardinalities):
            # Extract one-hot segment
            one_hot_segment = data[:, current_idx : current_idx + card]
            # Argmax to get class index
            class_indices = np.argmax(one_hot_segment, axis=1)
            # Clip safe
            class_indices = np.clip(class_indices, 0, len(self.label_encoders[col].classes_) - 1)
            # Inverse label encoding
            df_dict[col] = self.label_encoders[col].inverse_transform(class_indices)
            current_idx += card
            
        return pd.DataFrame(df_dict)[self.column_info['all_columns']]
    
    def save(self, path):
        """Save preprocessor state"""
        state = {
            'column_info': self.column_info,
            'categorical_cardinalities': self.categorical_cardinalities,
            'scalers': {col: {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}
                       for col, scaler in self.scalers.items()},
            'label_encoders': {col: le.classes_.tolist() 
                             for col, le in self.label_encoders.items()}
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self, path):
        """Load preprocessor state"""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.column_info = state['column_info']
        self.categorical_cols = state['column_info']['categorical']
        self.numerical_cols = state['column_info']['numerical']
        self.categorical_cardinalities = state.get('categorical_cardinalities', [])
        
        # Restore scalers
        for col, params in state['scalers'].items():
            scaler = StandardScaler()
            scaler.mean_ = np.array(params['mean'])
            scaler.scale_ = np.array(params['scale'])
            scaler.n_features_in_ = 1
            self.scalers[col] = scaler
        
        # Restore label encoders
        self.label_encoders = {}
        for col, classes in state['label_encoders'].items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            self.label_encoders[col] = le


class Trainer:
    """Trainer for Diffusion Transformer with CFG"""
    def __init__(
        self,
        model,
        diffusion,
        train_loader,
        val_loader=None,
        learning_rate=1e-4,
        device='cuda',
        save_dir='checkpoints',
        p_uncond=0.1
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.p_uncond = p_uncond
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """Train for one epoch with CFG"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            features, conditions = batch
            features = features.to(self.device)
            
            # Handle Conditions & CFG Masking
            if conditions is not None:
                conditions = conditions.to(self.device)
                # Randomly drop conditions for CFG training
                # Create mask: 1 = keep, 0 = drop (replace with null_class)
                mask = torch.rand(len(conditions), device=self.device) > self.p_uncond
                
                # Replace dropped conditions with null_class index
                # model.num_classes is the index for null class
                null_condition = torch.full_like(conditions, self.model.num_classes)
                conditions = torch.where(mask, conditions, null_condition)
            
            # Sample random timesteps
            batch_size = features.shape[0]
            t = torch.randint(
                0, self.diffusion.num_timesteps, (batch_size,), device=self.device
            ).long()
            
            # Compute loss
            loss = self.diffusion.p_losses(self.model, features, t, condition=conditions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return epoch_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        val_loss = 0.0
        
        for batch in self.val_loader:
            features, conditions = batch
            features = features.to(self.device)
            
            if conditions is not None:
                conditions = conditions.to(self.device)
                # No dropout during validation usually, unless testing unconditional performance
            
            batch_size = features.shape[0]
            t = torch.randint(
                0, self.diffusion.num_timesteps, (batch_size,), device=self.device
            ).long()
            
            loss = self.diffusion.p_losses(self.model, features, t, condition=conditions)
            val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
    
    def train(self, num_epochs, save_interval=10):
        """Train the model for multiple epochs"""
        print(f"Training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        # Save final model
        self.save_checkpoint('final')
        self.plot_losses()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'losses.png'))
        plt.close()


def train_diffusion_model(
    data_path,
    categorical_cols=None,
    condition_col=None, # Single column for simple Class Conditional Impl
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    num_timesteps=1000,
    batch_size=128,
    num_epochs=100,
    learning_rate=1e-4,
    val_split=0.1,
    p_uncond=0.1,
    device='cuda',
    save_dir='checkpoints'
):
    """
    Main training function
    """
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Separate condition column if present
    # We remove the condition column from the features to avoid leakage
    condition_data = None
    num_classes = None
    
    if condition_col is not None:
        print(f"Conditioning on: {condition_col}")
        # Encode condition
        le = LabelEncoder()
        condition_data = le.fit_transform(df[condition_col].astype(str))
        num_classes = len(le.classes_)
        
        # Save condition encoder
        os.makedirs(save_dir, exist_ok=True)
        import joblib
        joblib.dump(le, os.path.join(save_dir, 'condition_encoder.joblib'))
        
        # Drop condition col from training features
        # df = df.drop(columns=[condition_col])
        # Wait, usually we still want to generate the label? 
        # Standard Class Cond Gen: P(X | Y), we generate X. Y is given.
        # X usually includes all features EXCEPT Y.
        pass
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    # Note: We pass categorical_cols explicitly. 
    # If condition_col was dropped, ensure categorical_cols doesn't contain it
    if condition_col in categorical_cols:
        categorical_cols = [c for c in categorical_cols if c != condition_col]
        
    data_transformed = preprocessor.fit_transform(df.drop(columns=[condition_col] if condition_col else []), categorical_cols=categorical_cols)
    
    # Save preprocessor
    os.makedirs(save_dir, exist_ok=True)
    preprocessor.save(os.path.join(save_dir, 'preprocessor.json'))
    
    # Split data
    if val_split > 0:
        if condition_data is not None:
             train_data, val_data, train_cond, val_cond = train_test_split(
                data_transformed, condition_data, test_size=val_split, random_state=42
            )
        else:
            train_data, val_data = train_test_split(
                data_transformed, test_size=val_split, random_state=42
            )
            train_cond, val_cond = None, None
    else:
        train_data = data_transformed
        train_cond = condition_data
        val_data = None
        val_cond = None
    
    # Create datasets
    train_dataset = TabularDataset(train_data, condition=train_cond)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    val_loader = None
    if val_data is not None:
        val_dataset = TabularDataset(val_data, condition=val_cond)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
    
    # Create model
    print("Creating model...")
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
    
    # Create diffusion process
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        device=device,
        save_dir=save_dir,
        p_uncond=p_uncond
    )
    
    # Train
    trainer.train(num_epochs=num_epochs)
    
    return model, diffusion, preprocessor


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Diffusion Transformer')
    parser.add_argument('--data_path', type=str, default='data.csv', help='Path to dataset')
    parser.add_argument('--categorical_cols', type=str, nargs='+', default=None, help='List of categorical columns')
    parser.add_argument('--condition_col', type=str, default=None, help='Column to use for conditioning')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--p_uncond', type=float, default=0.1, help='Probability of unconditional training (CFG)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    # Handle categorical_cols being None if not provided
    # If not provided, preprocessor infers them (but we might want to be explicit)
    
    train_diffusion_model(
        data_path=args.data_path,
        categorical_cols=args.categorical_cols,
        condition_col=args.condition_col,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_timesteps=args.num_timesteps,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        p_uncond=args.p_uncond,
        device=args.device,
        save_dir=args.save_dir
    )
