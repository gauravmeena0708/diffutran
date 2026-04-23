import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DSDataset(Dataset):
    """
    Dataset for Dual-Stream Flow Matching.
    Stores preprocessed indices/values (CPU) to save memory.
    """
    def __init__(self, df, tokenizer, condition_col=None, label_encoder=None):
        """
        Args:
            df: Pandas DataFrame
            tokenizer: Fitted QuantileTokenizer
            condition_col: Name of column to use as condition (class label)
            label_encoder: Fitted LabelEncoder for condition
        """
        self.tokenizer = tokenizer
        
        # Handle Condition
        self.y = None
        if condition_col is not None and label_encoder is not None:
             y_data = df[condition_col].astype(str).values
             known_mask = np.isin(y_data, label_encoder.classes_)
             y_data[~known_mask] = label_encoder.classes_[0]
             y_indices = label_encoder.transform(y_data)
             self.y = torch.LongTensor(y_indices)
             
             if tokenizer.is_temporal:
                 # Reshape y to [Batch, Time]
                 B = len(self.y) // tokenizer.sequence_len
                 self.y = self.y.view(B, tokenizer.sequence_len)
                 # Optionally: Take last step label if model expects [Batch]
                 # For now, keep as [B, T] and let model handle if it wants [B]
                 # But our DualStreamTransformer expects [B] for class_embedder.
                 # Let's take the first step label for the whole sequence for now.
                 self.y = self.y[:, 0] 
        
        # Preprocess features
        self.x_num, self.x_cat = tokenizer.preprocess(df)
        
    def __len__(self):
        return len(self.x_num)
    
    def __getitem__(self, idx):
        # Return tuple (x_num, x_cat, y)
        if self.y is not None:
            return self.x_num[idx], self.x_cat[idx], self.y[idx]
        else:
            return self.x_num[idx], self.x_cat[idx], None

def get_dataloader(df, tokenizer, batch_size=512, shuffle=True, num_workers=4, condition_col=None, label_encoder=None):
    """
    Create a DataLoader for DS-Flow.
    """
    dataset = DSDataset(df, tokenizer, condition_col, label_encoder)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True 
    )
    return loader
