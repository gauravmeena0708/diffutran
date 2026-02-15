import torch
import torch.nn as nn

class TransformerModule(nn.Module):
    """
    Standard Transformer Encoder.
    """
    def __init__(self, d_model, nhead, num_layers, dropout=0.1, dim_feedforward=2048):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        """
        Args:
            src: [batch_size, seq_len, d_model]
        """
        return self.encoder(src)
