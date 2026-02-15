import torch
import torch.nn as nn
import math

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timestep encoding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class FeatureEmbedder(nn.Module):
    """
    Embeds numerical and categorical features into a common dimension.
    Used for Token-per-Feature architecture.
    """
    def __init__(self, num_numerical, categorical_cardinalities, d_model):
        """
        Args:
            num_numerical: Number of numerical features
            categorical_cardinalities: List of cardinalities for each categorical feature
            d_model: Embedding dimension
        """
        super().__init__()
        self.num_numerical = num_numerical
        self.categorical_cardinalities = categorical_cardinalities
        self.d_model = d_model

        # Numerical embeddings: Independent weights for each numerical feature
        # We project each scalar x_i to a vector of size d_model
        # Effectively: Weight_i * x_i + Bias_i
        self.numerical_proj = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_numerical)
        ])

        # Categorical embeddings]
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(card, d_model) for card in categorical_cardinalities
        ])
        
        # Learnable "CLS" token if needed, or we just use features
        # For this implementation, we'll just use the feature tokens

    def forward(self, x_num, x_cat):
        """
        Args:
            x_num: Numerical features [batch_size, num_numerical]
            x_cat: Categorical features [batch_size, num_categorical]
        
        Returns:
            embeddings: [batch_size, num_features, d_model]
        """
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        embeddings = []

        # Process numerical features
        if self.num_numerical > 0:
            for i in range(self.num_numerical):
                # Unsqueeze to [batch_size, 1] to apply Linear(1, d_model)
                emb = self.numerical_proj[i](x_num[:, i].unsqueeze(1))
                embeddings.append(emb.unsqueeze(1)) # [batch_size, 1, d_model]

        # Process categorical features
        if self.categorical_cardinalities:
            for i in range(len(self.categorical_cardinalities)):
                emb = self.categorical_embeddings[i](x_cat[:, i].long())
                embeddings.append(emb.unsqueeze(1)) # [batch_size, 1, d_model]

        # Concatenate all tokens
        return torch.cat(embeddings, dim=1)
