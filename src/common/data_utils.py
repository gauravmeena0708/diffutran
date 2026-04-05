import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

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
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            cardinality = len(le.classes_)
            self.categorical_cardinalities.append(cardinality)
        
        # Scale numerical columns
        for col in self.numerical_cols:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
            
        processed_data = []
        
        # 1. Numerical
        if self.numerical_cols:
            processed_data.append(df[self.numerical_cols].values)
            
        # 2. Categorical (One-Hot)
        for col in self.categorical_cols:
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
        
        processed_data = []
        
        # Numerical
        for col in self.numerical_cols:
            df[col] = self.scalers[col].transform(df[[col]])
        if self.numerical_cols:
             processed_data.append(df[self.numerical_cols].values)

        # Categorical
        for col in self.categorical_cols:
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
            one_hot_segment = data[:, current_idx : current_idx + card]
            # Argmax to get class index
            class_indices = np.argmax(one_hot_segment, axis=1)
            # Clip safe
            class_indices = np.clip(class_indices, 0, len(self.label_encoders[col].classes_) - 1)
            # Inverse label encoding
            df_dict[col] = self.label_encoders[col].inverse_transform(class_indices)
            current_idx += card
            
        return pd.DataFrame(df_dict)[self.column_info['all_columns']]
    
    def get_constraint_tensors(self, constraints, batch_size, device='cpu'):
        """
        Build constraint_x0 and constraint_mask tensors from a {col: value} dict.

        The flat tensor layout is:
          [num_0, num_1, ..., num_N-1 | cat_0 one-hot | cat_1 one-hot | ...]

        Returns
        -------
        constraint_x0   : FloatTensor [batch_size, total_dim]
            Constraint values in preprocessed (scaled / one-hot) space.
            Zero everywhere that is not constrained.
        constraint_mask : FloatTensor [batch_size, total_dim]
            1.0 for every element that belongs to a constrained column, else 0.0.
        """
        total_dim = len(self.numerical_cols) + sum(self.categorical_cardinalities)
        constraint_x0   = torch.zeros(batch_size, total_dim)
        constraint_mask = torch.zeros(batch_size, total_dim)

        for col, val in constraints.items():
            if col in self.numerical_cols:
                idx = self.numerical_cols.index(col)
                scaled = float(self.scalers[col].transform([[float(val)]])[0, 0])
                constraint_x0[:, idx]   = scaled
                constraint_mask[:, idx] = 1.0

            elif col in self.categorical_cols:
                offset = len(self.numerical_cols)
                for i, c in enumerate(self.categorical_cols):
                    card = self.categorical_cardinalities[i]
                    if c == col:
                        class_idx = int(
                            self.label_encoders[col].transform([str(val)])[0]
                        )
                        one_hot = torch.zeros(card)
                        one_hot[class_idx] = 1.0
                        constraint_x0[:, offset : offset + card]   = one_hot
                        constraint_mask[:, offset : offset + card] = 1.0
                        break
                    offset += card

        return constraint_x0.to(device), constraint_mask.to(device)

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
