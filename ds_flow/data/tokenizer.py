import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
import joblib

class QuantileTokenizer(nn.Module):
    """
    Handles preprocessing and tokenization for tabular data.
    - Fits QuantileTransformer for numerical features.
    - Fits LabelEncoders for categorical features.
    - Provides embeddings for categorical and linear projections for numerical.
    """
    def __init__(self, d_model=512, n_quantiles=1000, output_dist='normal', device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.n_quantiles = n_quantiles
        self.output_dist = output_dist
        self.device = device
        
        # State
        self.numerical_cols = []
        self.categorical_cols = []
        self.label_encoders = {}
        self.quantile_transformers = {}
        self.original_dtypes = {}
        
        # Embeddings (initialized during fit)
        self.cat_embeddings = nn.ModuleDict()
        self.num_projections = nn.ModuleDict()
        self.num_biases = nn.ParameterDict() # Bias for numerical projection
        
        # Decoders (for reconstruction)
        self.cat_decoders = nn.ModuleDict()
        self.num_decoders = nn.ModuleDict()
        
        # Metadata
        self.cardinalities = {}
        
    def fit(self, df, categorical_cols=None):
        """Fit transformers and initialize embedding layers"""
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.categorical_cols = categorical_cols
        self.numerical_cols = [c for c in df.columns if c not in categorical_cols]
        self.original_dtypes = df.dtypes.to_dict()
        
        # Fit Categorical
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            le.fit(df[col])
            self.label_encoders[col] = le
            cardinality = len(le.classes_)
            self.cardinalities[col] = cardinality
            
            # Encoder: Index -> d_model
            self.cat_embeddings[col] = nn.Embedding(cardinality, self.d_model)
            # Decoder: d_model -> Cardinality (Logits)
            self.cat_decoders[col] = nn.Linear(self.d_model, cardinality)
            
        # Fit Numerical
        for col in self.numerical_cols:
            qt = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, len(df)), 
                output_distribution=self.output_dist,
                subsample=100000
            )
            data = df[col].values.reshape(-1, 1)
            qt.fit(data)
            self.quantile_transformers[col] = qt
            
            # Encoder: 1 -> d_model
            self.num_projections[col] = nn.Linear(1, self.d_model)
            # Decoder: d_model -> 1 (Value)
            self.num_decoders[col] = nn.Linear(self.d_model, 1)
            
        return self

    def preprocess(self, df):
        """
        Convert DataFrame to Tensors (CPU side).
        Returns:
            x_num: Tensor [B, N_num] (Float)
            x_cat: Tensor [B, N_cat] (Long)
        """
        # Numerical
        if self.numerical_cols:
            x_nums = []
            for col in self.numerical_cols:
                qt = self.quantile_transformers[col]
                data = df[col].values.reshape(-1, 1)
                transformed = qt.transform(data) # [B, 1]
                x_nums.append(transformed)
            x_num = torch.FloatTensor(np.hstack(x_nums))
        else:
            x_num = torch.empty(len(df), 0)
            
        # Categorical
        if self.categorical_cols:
            x_cats = []
            for col in self.categorical_cols:
                le = self.label_encoders[col]
                data = df[col].astype(str).values
                indices = le.transform(data).reshape(-1, 1)
                x_cats.append(indices)
            x_cat = torch.LongTensor(np.hstack(x_cats))
        else:
            x_cat = torch.empty(len(df), 0, dtype=torch.long)
            
        return x_num, x_cat

    def forward(self, x_num, x_cat):
        """
        Embed tensors into Tokens [B, N_cols, D] (GPU side).
        """
        all_embeddings = []
        
        # Process Numerical
        for i, col in enumerate(self.numerical_cols):
            # x_num[:, i] is [B]
            tensor = x_num[:, i].unsqueeze(1).to(self.device) # [B, 1]
            embedding = self.num_projections[col](tensor).unsqueeze(1) # [B, 1, D]
            all_embeddings.append(embedding)
            
        # Process Categorical
        for i, col in enumerate(self.categorical_cols):
            tensor = x_cat[:, i].to(self.device) # [B]
            embedding = self.cat_embeddings[col](tensor).unsqueeze(1) # [B, 1, D]
            all_embeddings.append(embedding)
            
        # Concatenate: [Num..., Cat...]
        x_tokens = torch.cat(all_embeddings, dim=1) 
        return x_tokens

    def transform(self, df):
        """High-level helper: DataFrame -> Tokens"""
        x_num, x_cat = self.preprocess(df)
        return self.forward(x_num, x_cat)


    def decode(self, x_tokens):
        """
        Decode tokens back to values/logits.
        Args:
            x_tokens: [B, N_cols, D]
        Returns:
            decoded_dict: {col: Tensor}
                - Numerical cols: [B, 1] (Continuous value in quantile space)
                - Categorical cols: [B, Card] (Logits)
        """
        decoded = {}
        current_idx = 0
        
        # 1. Numerical
        for col in self.numerical_cols:
            # Slice token
            token = x_tokens[:, current_idx, :] # [B, D]
            # Decode
            val = self.num_decoders[col](token) # [B, 1]
            decoded[col] = val
            current_idx += 1
            
        # 2. Categorical
        for col in self.categorical_cols:
            token = x_tokens[:, current_idx, :] # [B, D]
            logits = self.cat_decoders[col](token) # [B, Card]
            decoded[col] = logits
            current_idx += 1
            
        return decoded

    def inverse_transform(self, x_tokens):
        """
        Reconstruct DataFrame from Tokens [B, N, D] (Inference Mode).
        Uses argmax for categories and direct value for numericals.
        """
        decoded = self.decode(x_tokens)
        df_dict = {}
        
        for col in self.numerical_cols:
            val_tensor = decoded[col].detach().cpu().numpy() # [B, 1]
            qt = self.quantile_transformers[col]
            # Inverse Quantile Transform
            # Clip to be safe within normal distribution limits approx [-5, 5]? 
            # QT handles it.    
            original_val = qt.inverse_transform(val_tensor)
            df_dict[col] = original_val.flatten()
            
        for col in self.categorical_cols:
            logits = decoded[col].detach().cpu().numpy() # [B, Card]
            indices = np.argmax(logits, axis=1)
            le = self.label_encoders[col]
            original_val = le.inverse_transform(indices)
            df_dict[col] = original_val
            
        return pd.DataFrame(df_dict)

    
    def recover_dataframe(self, num_vals, cat_indices):
        """
        Convert raw continuous values (quantile space) and cat indices back to DataFrame.
        """
        df_dict = {}
        
        # Numerical
        for i, col in enumerate(self.numerical_cols):
            qt = self.quantile_transformers[col]
            # num_vals[i] is [B, 1] tensor
            vals = num_vals[i].detach().cpu().numpy()
            inv_vals = qt.inverse_transform(vals)
            df_dict[col] = inv_vals.flatten()
            
        # Categorical
        for i, col in enumerate(self.categorical_cols):
            le = self.label_encoders[col]
            # cat_indices[i] is [B] tensor
            idxs = cat_indices[i].detach().cpu().numpy()
            # Clip
            idxs = np.clip(idxs, 0, len(le.classes_)-1)
            inv_vals = le.inverse_transform(idxs)
            df_dict[col] = inv_vals
            
        return pd.DataFrame(df_dict)

    def save(self, path):
         joblib.dump(self, path)
    
    @staticmethod
    def load(path):
        return joblib.load(path)
