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
             # Drop condition from input features?
             # Usually standard practice is to drop target/condition from features if it's the target
             # But here we pass full DF to preprocess.
             # We should probably handle this outside, but let's assume DF passed to tokenizer 
             # ALREADY had condition_col dropped OR we just extract it here?
             # If tokenizer was fit on DF *with* condition_col, it will treat it as a feature.
             # We must ensure tokenizer is fit on X (without y).
             # We will handle this in train.py.
             # Here we assume df has y.
             y_data = df[condition_col].astype(str).values
             self.y = torch.LongTensor(label_encoder.transform(y_data))
        
        # Preprocess features (df should ideally not contain condition_col if we don't want it as feature)
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
