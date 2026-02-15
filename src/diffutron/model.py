import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .embeddings import SinusoidalPositionalEmbedding

class DiffusionModel(nn.Module):
    def __init__(
        self,
        num_numerical_features,
        categorical_cardinalities,
        d_model=256,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        dim_feedforward=1024,
        num_classes=None # For conditional generation (class conditioning)
    ):
        super().__init__()
        self.d_model = d_model
        self.num_numerical = num_numerical_features
        self.categorical_cardinalities = categorical_cardinalities
        self.num_categorical = len(categorical_cardinalities)
        self.num_classes = num_classes

        # Dimensions
        # Numerical features are scalars (dim 1)
        # Categorical features are one-hot vectors (dim cardinality)
        self.numerical_dims = [1] * num_numerical_features
        self.categorical_dims = categorical_cardinalities
        self.feature_dims = self.numerical_dims + self.categorical_dims
        self.num_features = len(self.feature_dims)

        # 1. Feature Projections (Input -> d_model)
        # We project each feature (scalar or one-hot vector) to d_model
        self.input_projections = nn.ModuleList()
        for dim in self.feature_dims:
            self.input_projections.append(nn.Linear(dim, d_model))

        # 2. Class Embedding for CFG
        if num_classes is not None:
             # +1 for "null" class (unconditional)
            self.class_embedding = nn.Embedding(num_classes + 1, d_model)

        # 3. Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 4. Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Output Heads (d_model -> Feature Dim)
        self.output_projections = nn.ModuleList()
        for dim in self.feature_dims:
            self.output_projections.append(nn.Linear(d_model, dim))

    def forward(self, x, timesteps, condition=None):
        """
        Args:
            x: [batch_size, total_input_dim] (Concatenated noisy features)
            timesteps: [batch_size]
            condition: [batch_size] (Class labels 0..K-1) or None for unconditional
        
        Returns:
            predicted_noise: [batch_size, total_input_dim]
        """
        batch_size = x.shape[0]
        device = x.device

        # 1. Split input into features
        # We need to slice x based on feature_dims
        feature_slices = []
        current_idx = 0
        for dim in self.feature_dims:
            feature_slices.append(x[:, current_idx : current_idx + dim])
            current_idx += dim
            
        # 2. Project Features to Tokens
        tokens = []
        for i, feature_slice in enumerate(feature_slices):
            # [batch, dim] -> [batch, d_model]
            emb = self.input_projections[i](feature_slice)
            tokens.append(emb.unsqueeze(1)) # [batch, 1, d_model]
            
        # 3. Concatenate Feature Tokens
        x_emb = torch.cat(tokens, dim=1) # [batch, num_features, d_model]
        
        # 4. Add Class Token (Condition)
        if self.num_classes is not None:
            if condition is None:
                 # Null condition (index = num_classes)
                 # We create a specific index for "null"
                 y = torch.full((batch_size,), self.num_classes, device=device, dtype=torch.long)
            else:
                # Assuming condition is class index tensor
                y = condition
            
            y_emb = self.class_embedding(y) # [batch, d_model]
            x_emb = torch.cat([y_emb.unsqueeze(1), x_emb], dim=1) # [batch, num_features + 1, d_model]

        # 5. Add Time Embedding
        t_emb = self.time_embed(timesteps) # [batch, d_model]
        # Broadcast time embedding to all tokens
        x_emb = x_emb + t_emb.unsqueeze(1)
        
        # 6. Transformer Forward
        h = self.transformer(x_emb)
        
        # 7. Output Projection
        outputs = []
        # If conditional, skip the first token (class token)
        start_token_idx = 1 if self.num_classes is not None else 0
        
        for i in range(self.num_features):
            token_h = h[:, start_token_idx + i, :] # [batch, d_model]
            out = self.output_projections[i](token_h) # [batch, dim]
            outputs.append(out)
            
        # 8. Concatenate Outputs
        return torch.cat(outputs, dim=1)
