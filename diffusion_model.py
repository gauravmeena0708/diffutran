import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer for tabular data generation.
    Supports conditional generation based on specified features.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        time_embed_dim=128,
        condition_dim=None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        
        # Time embedding
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Condition projection (if using conditional generation)
        if condition_dim is not None:
            self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # Time projection for adaptive layer norm
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim * 2)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, timesteps, condition=None):
        """
        Args:
            x: Input tensor [batch_size, input_dim]
            timesteps: Timestep tensor [batch_size]
            condition: Optional condition tensor [batch_size, condition_dim]
        
        Returns:
            Predicted noise [batch_size, input_dim]
        """
        batch_size = x.shape[0]
        
        # Time embedding
        t_emb = self.time_mlp(timesteps)  # [batch_size, time_embed_dim]
        
        # Project input to hidden dimension
        h = self.input_proj(x)  # [batch_size, hidden_dim]
        
        # Add condition if provided
        if condition is not None and self.condition_dim is not None:
            cond_emb = self.condition_proj(condition)
            h = h + cond_emb
        
        # Reshape for transformer: [batch_size, 1, hidden_dim]
        h = h.unsqueeze(1)
        
        # Apply transformer blocks
        for block in self.blocks:
            h = block(h)
        
        # Remove sequence dimension
        h = h.squeeze(1)  # [batch_size, hidden_dim]
        
        # Apply time-conditioned adaptive layer norm
        time_scale_shift = self.time_proj(t_emb)
        scale, shift = time_scale_shift.chunk(2, dim=1)
        h = self.norm_out(h) * (1 + scale) + shift
        
        # Output projection
        output = self.output_proj(h)
        
        return output


class GaussianDiffusion:
    """
    Gaussian Diffusion process for training and sampling.
    Uses DDPM (Denoising Diffusion Probabilistic Models) formulation.
    """
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_type='linear'
    ):
        self.num_timesteps = num_timesteps
        
        # Define beta schedule
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in IDDPM paper"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to data according to the schedule
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, arr, timesteps, broadcast_shape):
        """Extract values from array at timesteps and reshape"""
        arr = arr.to(timesteps.device)
        res = arr.gather(0, timesteps)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def p_losses(self, model, x_start, t, condition=None, noise=None):
        """
        Training loss: predict noise added to x_0 to get x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward process
        x_noisy = self.q_sample(x_start, t, noise=noise)
        
        # Predict noise
        predicted_noise = model(x_noisy, t, condition=condition)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, condition=None):
        """
        Reverse diffusion process: sample x_{t-1} from x_t
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
        
        # Predict noise
        predicted_noise = model(x, t, condition=condition)
        
        # Mean of the reverse process
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, condition=None, device='cuda'):
        """
        Generate samples from the model
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, condition=condition)
        
        return x
