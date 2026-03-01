import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for training and sampling.
    """
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_type='linear'
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Define beta schedule
        if schedule_type == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        self.register_buffer('betas', betas)

        # Pre-compute diffusion parameters
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / self.alphas))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance', 
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
        if arr.device != timesteps.device:
            arr = arr.to(timesteps.device)
        res = arr.gather(0, timesteps)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def p_losses(self, model, x_start, t, condition=None, noise=None):
        """
        Training loss: predict noise added to x_0 to get x_t
        """
        # x_start is [batch, num_features] (concatenated numerical + continuous encodings of categorical)
        # OR it handles list of features.
        # For simplicity, let's assume `x_start` is the flattened vector of all features
        # normalized/processed to be continuous.
        # This matches standard TabDDPM.
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward process
        x_noisy = self.q_sample(x_start, t, noise=noise)
        
        # Predict noise
        # The model needs to handle splitting x_noisy into tokens if it's Token-per-Feature
        # We assume the model wrapper handles this splitting.
        predicted_noise = model(x_noisy, t, condition=condition)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, condition=None, guidance_scale=1.0):
        """
        Reverse diffusion process: sample x_{t-1} from x_t
        Supports Classifier-Free Guidance (CFG)
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict noise
        if guidance_scale == 1.0 or condition is None:
            predicted_noise = model(x, t, condition=condition)
        else:
            # CFG: epsilon = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
            # We assume condition=None triggers unconditional generation in the model
            noise_cond = model(x, t, condition=condition)
            noise_uncond = model(x, t, condition=None)
            
            predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
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
    def sample(self, model, shape, condition=None, guidance_scale=1.0, device='cuda'):
        """
        Generate samples from the model
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(
                model, x, t, condition=condition, guidance_scale=guidance_scale
            )
        
        return x
