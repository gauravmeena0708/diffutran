import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
import re

class QuantileTokenizer(nn.Module):
    """
    Handles preprocessing and tokenization for tabular data.
    - Fits QuantileTransformer for numerical features (with robust scaling jitter).
    - Fits LabelEncoders for categorical features (with Hash Fallback for high cardinality).
    - Handles np.nan missing values natively via categorical indicators.
    """
    def __init__(self, d_model=512, n_quantiles=1000, output_dist='normal', max_cardinality=100):
        super().__init__()
        self.d_model = d_model
        self.n_quantiles = n_quantiles
        self.output_dist = output_dist
        self.max_cardinality = max_cardinality
        self.register_buffer('_device_tracker', torch.zeros(1))
        
        # State
        self.numerical_cols = []
        self.categorical_cols = []
        self.label_encoders = {}
        self.quantile_transformers = {}
        self.original_dtypes = {}
        self.is_hashed = {}
        self.missing_indicators = {} # num_col -> cat_col_name
        
        # Embeddings (initialized during fit)
        self.cat_embeddings = nn.ModuleDict()
        self.num_projections = nn.ModuleDict()
        self.cat_decoders = nn.ModuleDict()
        self.num_decoders = nn.ModuleDict()
        
        # Metadata
        self.cardinalities = {}

    @staticmethod
    def _module_key(col: str) -> str:
        """Return a stable PyTorch-safe key for ModuleDict storage."""
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(col))
        safe = re.sub(r"_+", "_", safe).strip("_")
        return f"col_{safe or 'unnamed'}"

    @property
    def device(self):
        return self._device_tracker.device

    def fit(self, df, categorical_cols=None):
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
        base_num_cols = [c for c in df.columns if c not in categorical_cols]
        base_cat_cols = list(categorical_cols)
        
        self.original_dtypes = df.dtypes.to_dict()
        
        # 1. Missing Value Indicators for Numerical
        for col in base_num_cols:
            if df[col].isna().any():
                indicator_col = f"{col}_is_missing"
                self.missing_indicators[col] = indicator_col
                base_cat_cols.append(indicator_col)
                df[indicator_col] = df[col].isna().astype(str)
                # Impute numerical with 0.0 for fitting
                df[col] = df[col].fillna(0.0)
                
        self.numerical_cols = base_num_cols
        self.categorical_cols = base_cat_cols
        
        # 2. Fit Categorical
        for col in self.categorical_cols:
            df[col] = df[col].fillna('<MISSING>').astype(str)
            unique_vals = df[col].unique()
            
            if len(unique_vals) > self.max_cardinality:
                self.is_hashed[col] = True
                cardinality = self.max_cardinality
            else:
                self.is_hashed[col] = False
                le = LabelEncoder()
                le.fit(unique_vals)
                self.label_encoders[col] = le
                cardinality = len(le.classes_)
                
            self.cardinalities[col] = cardinality
            key = self._module_key(col)
            self.cat_embeddings[key] = nn.Embedding(cardinality, self.d_model)
            self.cat_decoders[key] = nn.Linear(self.d_model, cardinality)
            
        # 3. Fit Numerical
        for col in self.numerical_cols:
            qt = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, len(df)), 
                output_distribution=self.output_dist,
                subsample=100000
            )
            data = df[col].values.astype(np.float32) # Already imputed
            
            # Robust scaling: add tiny jitter to prevent identical outlier collapse
            if len(np.unique(data)) < 0.1 * len(data):
                jitter = np.random.normal(0, 1e-6, size=data.shape)
                data = data + jitter
                
            qt.fit(data.reshape(-1, 1))
            self.quantile_transformers[col] = qt

            key = self._module_key(col)
            self.num_projections[key] = nn.Linear(1, self.d_model)
            self.num_decoders[key] = nn.Linear(self.d_model, 1)
            
        return self

    def preprocess(self, df):
        df = df.copy()
        
        if self.numerical_cols:
            x_nums = []
            for col in self.numerical_cols:
                if col in self.missing_indicators:
                    indicator = self.missing_indicators[col]
                    if indicator not in df.columns:
                        df[indicator] = df[col].isna().astype(str)
                df[col] = df[col].fillna(0.0)
                
                qt = self.quantile_transformers[col]
                data = df[col].values.astype(np.float32).reshape(-1, 1)
                transformed = qt.transform(data) 
                x_nums.append(transformed)
            x_num = torch.FloatTensor(np.hstack(x_nums))
        else:
            x_num = torch.empty(len(df), 0)
            
        if self.categorical_cols:
            x_cats = []
            for col in self.categorical_cols:
                if col not in df.columns:
                    # Should only happen if user didn't provide indicator
                    df[col] = '<MISSING>'
                data = df[col].fillna('<MISSING>').astype(str).values
                
                if self.is_hashed[col]:
                    # Deterministic hash
                    indices = np.array([hash(str(x)) % self.max_cardinality for x in data]).reshape(-1, 1)
                else:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    known_mask = np.isin(data, le.classes_)
                    data[~known_mask] = '<MISSING>' if '<MISSING>' in le.classes_ else le.classes_[0]
                    indices = le.transform(data).reshape(-1, 1)
                x_cats.append(indices)
            x_cat = torch.LongTensor(np.hstack(x_cats))
        else:
            x_cat = torch.empty(len(df), 0, dtype=torch.long)
            
        return x_num, x_cat

    def forward(self, x_num, x_cat):
        all_embeddings = []
        for i, col in enumerate(self.numerical_cols):
            tensor = x_num[:, i].unsqueeze(1).to(self.device)
            embedding = self.num_projections[self._module_key(col)](tensor).unsqueeze(1)
            all_embeddings.append(embedding)
            
        for i, col in enumerate(self.categorical_cols):
            tensor = x_cat[:, i].to(self.device) 
            embedding = self.cat_embeddings[self._module_key(col)](tensor).unsqueeze(1)
            all_embeddings.append(embedding)
            
        return torch.cat(all_embeddings, dim=1) if all_embeddings else torch.empty(x_num.shape[0], 0, self.d_model, device=self.device)

    def transform(self, df):
        x_num, x_cat = self.preprocess(df)
        return self.forward(x_num, x_cat)

    def decode(self, x_tokens):
        decoded = {}
        current_idx = 0
        for col in self.numerical_cols:
            token = x_tokens[:, current_idx, :]
            val = self.num_decoders[self._module_key(col)](token)
            decoded[col] = val
            current_idx += 1
            
        for col in self.categorical_cols:
            token = x_tokens[:, current_idx, :]
            logits = self.cat_decoders[self._module_key(col)](token)
            decoded[col] = logits
            current_idx += 1
        return decoded

    def inverse_transform(self, x_tokens):
        decoded = self.decode(x_tokens)
        df_dict = {}
        
        for col in self.numerical_cols:
            val_tensor = decoded[col].detach().cpu().numpy()
            qt = self.quantile_transformers[col]
            # Clip slightly to avoid inf
            val_tensor = np.clip(val_tensor, -5.0, 5.0)
            original_val = qt.inverse_transform(val_tensor)
            df_dict[col] = original_val.flatten()
            
        for col in self.categorical_cols:
            logits = decoded[col].detach().cpu().numpy()
            indices = np.argmax(logits, axis=1)
            
            if self.is_hashed[col]:
                df_dict[col] = [f"<HASH_{idx}>" for idx in indices]
            else:
                le = self.label_encoders[col]
                df_dict[col] = le.inverse_transform(indices)
                
        # Handle Missing Indicators
        for col, ind_col in self.missing_indicators.items():
            mask = (df_dict[ind_col] == 'True')
            df_dict[col] = np.where(mask, np.nan, df_dict[col])
            del df_dict[ind_col] # Remove indicator from final output
            
        return pd.DataFrame(df_dict)

    def recover_dataframe(self, num_vals, cat_indices):
        df_dict = {}
        for i, col in enumerate(self.numerical_cols):
            qt = self.quantile_transformers[col]
            vals = num_vals[i].detach().cpu().numpy()
            vals = np.clip(vals, -5.0, 5.0)
            inv_vals = qt.inverse_transform(vals)
            df_dict[col] = inv_vals.flatten()
            
        for i, col in enumerate(self.categorical_cols):
            idxs = cat_indices[i].detach().cpu().numpy()
            if self.is_hashed[col]:
                inv_vals = [f"<HASH_{idx}>" for idx in idxs]
            else:
                le = self.label_encoders[col]
                idxs = np.clip(idxs, 0, len(le.classes_)-1)
                inv_vals = le.inverse_transform(idxs)
            df_dict[col] = inv_vals
            
        for col, ind_col in self.missing_indicators.items():
            mask = (df_dict[ind_col] == 'True')
            df_dict[col] = np.where(mask, np.nan, df_dict[col])
            del df_dict[ind_col]
            
        return pd.DataFrame(df_dict)

    def save(self, path):
        state_dict = self.state_dict()
        metadata = {
            'd_model': self.d_model,
            'n_quantiles': self.n_quantiles,
            'output_dist': self.output_dist,
            'max_cardinality': self.max_cardinality,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'cardinalities': self.cardinalities,
            'is_hashed': self.is_hashed,
            'missing_indicators': self.missing_indicators,
            'label_encoders': {col: le.classes_.tolist() for col, le in self.label_encoders.items()},
            'quantile_transformers': {
                col: {
                    'quantiles_': qt.quantiles_.tolist(),
                    'references_': qt.references_.tolist(),
                    'n_quantiles_': qt.n_quantiles_,
                    'output_distribution': qt.output_distribution,
                } for col, qt in self.quantile_transformers.items()
            }
        }
        torch.save({'state_dict': state_dict, 'metadata': metadata}, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location=device, weights_only=True)
        metadata = data['metadata']
        
        instance = cls(
            d_model=metadata['d_model'],
            n_quantiles=metadata['n_quantiles'],
            output_dist=metadata['output_dist'],
            max_cardinality=metadata.get('max_cardinality', 100)
        )
        instance.numerical_cols = metadata['numerical_cols']
        instance.categorical_cols = metadata['categorical_cols']
        instance.cardinalities = metadata['cardinalities']
        instance.is_hashed = metadata.get('is_hashed', {c: False for c in instance.categorical_cols})
        instance.missing_indicators = metadata.get('missing_indicators', {})
        
        for col, classes in metadata['label_encoders'].items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            instance.label_encoders[col] = le
            
        for col, qt_state in metadata['quantile_transformers'].items():
            qt = QuantileTransformer(n_quantiles=len(qt_state['quantiles_']), output_distribution=qt_state['output_distribution'])
            qt.quantiles_ = np.array(qt_state['quantiles_'])
            qt.references_ = np.array(qt_state['references_'])
            qt.n_quantiles_ = qt_state.get('n_quantiles_', qt.quantiles_.shape[0])
            instance.quantile_transformers[col] = qt
            
        for col in instance.categorical_cols:
            c = instance.cardinalities[col]
            key = instance._module_key(col)
            instance.cat_embeddings[key] = nn.Embedding(c, instance.d_model)
            instance.cat_decoders[key] = nn.Linear(instance.d_model, c)
            
        for col in instance.numerical_cols:
            key = instance._module_key(col)
            instance.num_projections[key] = nn.Linear(1, instance.d_model)
            instance.num_decoders[key] = nn.Linear(instance.d_model, 1)
            
        instance.load_state_dict(data['state_dict'])
        instance.to(device)
        return instance

if __name__ == "__main__":
    df = pd.DataFrame({
        'n1': [1.0, 2.0, np.nan, 4.0],
        'c1': ['A', 'B', 'A', np.nan],
        'c_high': [str(i) for i in range(4)]
    })
    tok = QuantileTokenizer(d_model=16, max_cardinality=2, n_quantiles=2)
    tok.fit(df)
    assert 'n1_is_missing' in tok.categorical_cols
    assert tok.is_hashed['c_high'] == True
    
    x_n, x_c = tok.preprocess(df)
    tokens = tok.forward(x_n, x_c)
    assert tokens.shape == (4, 4, 16) # 1 num + 3 cat
    
    dec = tok.inverse_transform(tokens)
    assert 'n1_is_missing' not in dec.columns
    assert 'n1' in dec.columns
    print("QuantileTokenizer Verification Passed!")
