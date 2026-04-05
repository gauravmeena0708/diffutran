import pytest
import math
import sys
import os
import torch
from unittest.mock import MagicMock

# Add project root to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.diffutron.diffusion import GaussianDiffusion


def test_q_sample_shape():
    num_timesteps = 100
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps)

    batch_size = 4
    num_features = 8
    x_start = torch.randn(batch_size, num_features)
    t = torch.randint(0, num_timesteps, (batch_size,))

    x_noisy = diffusion.q_sample(x_start, t)

    assert x_noisy.shape == x_start.shape

def test_q_sample_deterministic():
    num_timesteps = 10
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps, schedule_type='linear', beta_start=0.1, beta_end=0.2)

    batch_size = 2
    num_features = 3
    x_start = torch.ones(batch_size, num_features)
    t = torch.tensor([0, 5])
    noise = torch.ones(batch_size, num_features) * 0.5

    x_noisy = diffusion.q_sample(x_start, t, noise=noise)

    # Manual calculation
    betas = torch.linspace(0.1, 0.2, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    expected_0 = sqrt_alphas_cumprod[0] * x_start[0] + sqrt_one_minus_alphas_cumprod[0] * noise[0]
    expected_5 = sqrt_alphas_cumprod[5] * x_start[1] + sqrt_one_minus_alphas_cumprod[5] * noise[1]

    expected = torch.stack([expected_0, expected_5])

    assert torch.allclose(x_noisy, expected)

def test_q_sample_t0():
    # At t=0, x_noisy should be very close to x_start if beta_start is small
    num_timesteps = 1000
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps, beta_start=1e-8)

    batch_size = 10
    num_features = 20
    x_start = torch.randn(batch_size, num_features)
    t = torch.zeros(batch_size, dtype=torch.long)

    x_noisy = diffusion.q_sample(x_start, t)

    assert torch.allclose(x_noisy, x_start, atol=1e-3)

def test_q_sample_t_max():
    # At t=max, x_noisy should be mostly noise if alphas_cumprod is close to 0
    num_timesteps = 100
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps, beta_start=0.1, beta_end=0.9)

    batch_size = 1000
    num_features = 1
    x_start = torch.ones(batch_size, num_features) * 10.0
    t = torch.full((batch_size,), num_timesteps - 1, dtype=torch.long)

    x_noisy = diffusion.q_sample(x_start, t)

    alphas = 1.0 - torch.linspace(0.1, 0.9, num_timesteps)
    alphas_cumprod_max = torch.prod(alphas)

    expected_mean = math.sqrt(alphas_cumprod_max.item()) * 10.0

    assert torch.abs(x_noisy.mean() - expected_mean) < 0.1
    assert torch.abs(x_noisy.std() - 1.0) < 0.1

def test_q_sample_no_noise():
    num_timesteps = 100
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps)
    x_start = torch.randn(4, 8)
    t = torch.randint(0, num_timesteps, (4,))

    x_noisy = diffusion.q_sample(x_start, t)
    assert x_noisy.shape == x_start.shape

def test_q_sample_consistency():
    num_timesteps = 100
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps)
    x_start = torch.randn(4, 8)
    t = torch.randint(0, num_timesteps, (4,))
    noise = torch.randn_like(x_start)

    x_noisy1 = diffusion.q_sample(x_start, t, noise=noise)
    x_noisy2 = diffusion.q_sample(x_start, t, noise=noise)

    assert torch.equal(x_noisy1, x_noisy2)

def test_gaussian_diffusion_invalid_schedule_type():
    """Test that GaussianDiffusion raises ValueError for invalid schedule_type"""
    with pytest.raises(ValueError, match="Unknown schedule type: invalid_type"):
        GaussianDiffusion(schedule_type="invalid_type")

def test_gaussian_diffusion_valid_schedule_types():
    """Test that GaussianDiffusion accepts valid schedule types"""
    try:
        GaussianDiffusion(schedule_type="linear")
        GaussianDiffusion(schedule_type="cosine")
    except Exception as e:
        pytest.fail(f"Valid schedule types raised exception: {e}")

@pytest.fixture
def mock_model():
    """Creates a mock model that returns a tensor of the same shape as input."""
    model = MagicMock()
    model.side_effect = lambda x, t, **kwargs: torch.zeros_like(x)
    return model

def test_p_sample_shape(mock_model):
    """Test the output shape of the p_sample method."""
    diffusion = GaussianDiffusion(num_timesteps=10)
    x = torch.randn(2, 4)
    t = torch.tensor([5, 5])
    
    x_denoised = diffusion.p_sample(mock_model, x, t)
    
    assert x_denoised.shape == x.shape

def test_p_sample_t0(mock_model):
    """Test p_sample at t=0, which should be deterministic without noise."""
    diffusion = GaussianDiffusion(num_timesteps=10)
    x = torch.randn(2, 4)
    t = torch.tensor([0, 0])
    
    # At t=0, the model should ideally return the denoised image, and p_sample adds no noise
    x_denoised = diffusion.p_sample(mock_model, x, t)
    
    # With a mock model that predicts zero noise, the result should be related to x
    # in a predictable way. Let's verify it's not random.
    x_denoised_2 = diffusion.p_sample(mock_model, x, t)
    assert torch.allclose(x_denoised, x_denoised_2)

def test_sample_shape(mock_model):
    """Test the output shape of the full sample method."""
    diffusion = GaussianDiffusion(num_timesteps=10)
    shape = (3, 6)
    
    # Mock torch.full to work with MagicMock model
    original_full = torch.full
    torch.full = lambda *args, **kwargs: original_full(*args, **kwargs)
    
    generated_samples = diffusion.sample(mock_model, shape, device='cpu')
    
    assert generated_samples.shape == shape
    
    # Restore original torch.full
    torch.full = original_full


def test_sample_fast_inference_steps_shape(mock_model):
    """Test that num_inference_steps produces output with the correct shape."""
    diffusion = GaussianDiffusion(num_timesteps=100)
    shape = (4, 8)

    generated = diffusion.sample(
        mock_model, shape, device='cpu', num_inference_steps=10
    )

    assert generated.shape == shape


def test_sample_fast_inference_steps_fewer_calls(mock_model):
    """With num_inference_steps=5, the model should be called exactly 5 times."""
    num_timesteps = 100
    num_inference_steps = 5
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps)
    shape = (2, 6)

    diffusion.sample(
        mock_model, shape, device='cpu',
        num_inference_steps=num_inference_steps
    )

    assert mock_model.call_count == num_inference_steps


def test_sample_full_steps_when_none(mock_model):
    """When num_inference_steps=None, all timesteps are used."""
    num_timesteps = 20
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps)
    shape = (2, 4)

    diffusion.sample(mock_model, shape, device='cpu', num_inference_steps=None)

    assert mock_model.call_count == num_timesteps
