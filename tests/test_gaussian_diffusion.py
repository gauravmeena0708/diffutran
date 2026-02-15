import pytest
import math

def test_q_sample_shape():
    import torch
    from models.gaussian_diffusion import GaussianDiffusion
    num_timesteps = 100
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps)

    batch_size = 4
    num_features = 8
    x_start = torch.randn(batch_size, num_features)
    t = torch.randint(0, num_timesteps, (batch_size,))

    x_noisy = diffusion.q_sample(x_start, t)

    assert x_noisy.shape == x_start.shape

def test_q_sample_deterministic():
    import torch
    from models.gaussian_diffusion import GaussianDiffusion
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
    import torch
    from models.gaussian_diffusion import GaussianDiffusion
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
    import torch
    from models.gaussian_diffusion import GaussianDiffusion
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
    import torch
    from models.gaussian_diffusion import GaussianDiffusion
    num_timesteps = 100
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps)
    x_start = torch.randn(4, 8)
    t = torch.randint(0, num_timesteps, (4,))

    x_noisy = diffusion.q_sample(x_start, t)
    assert x_noisy.shape == x_start.shape

def test_q_sample_consistency():
    import torch
    from models.gaussian_diffusion import GaussianDiffusion
    num_timesteps = 100
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps)
    x_start = torch.randn(4, 8)
    t = torch.randint(0, num_timesteps, (4,))
    noise = torch.randn_like(x_start)

    x_noisy1 = diffusion.q_sample(x_start, t, noise=noise)
    x_noisy2 = diffusion.q_sample(x_start, t, noise=noise)

    assert torch.equal(x_noisy1, x_noisy2)
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Helper to mock dependencies if missing
try:
    import torch
except ImportError:
    # Create mocks for torch
    mock_torch = MagicMock()

    # Mock nn.Module specifically as it is used as base class
    class MockModule:
        def __init__(self):
            # Simulate nn.Module behavior of tracking buffers/params
            self._buffers = {}
            self._modules = {}

        def register_buffer(self, name, tensor):
            # Essential: set the attribute so it can be accessed later
            setattr(self, name, tensor)
            self._buffers[name] = tensor

        def __call__(self, *args, **kwargs):
            return MagicMock()

        def to(self, device): return self
        def eval(self): return self
        def train(self): return self
        def state_dict(self): return {}
        def load_state_dict(self, state_dict): pass

    mock_torch.nn = MagicMock()
    mock_torch.nn.Module = MockModule
    mock_torch.nn.functional = MagicMock()

    # Mock tensor operations used in init
    # linspace returns a mock that supports basic ops
    mock_tensor = MagicMock()
    # Support subtraction (1.0 - self.betas)
    mock_tensor.__rsub__.return_value = mock_tensor
    mock_tensor.__sub__.return_value = mock_tensor
    # Support sqrt
    mock_torch.sqrt.return_value = mock_tensor
    # Support cumprod
    mock_torch.cumprod.return_value = mock_tensor
    # Support linspace
    mock_torch.linspace.return_value = mock_tensor
    # Support cos (for cosine schedule)
    mock_torch.cos.return_value = mock_tensor
    # Support clip
    mock_torch.clip.return_value = mock_tensor

    sys.modules['torch'] = mock_torch
    sys.modules['torch.nn'] = mock_torch.nn
    sys.modules['torch.nn.functional'] = mock_torch.nn.functional

try:
    import numpy
except ImportError:
    sys.modules['numpy'] = MagicMock()

# Import the module under test
# This must be done AFTER mocking
from models.gaussian_diffusion import GaussianDiffusion

def test_gaussian_diffusion_invalid_schedule_type():
    """Test that GaussianDiffusion raises ValueError for invalid schedule_type"""
    with pytest.raises(ValueError, match="Unknown schedule type: invalid_type"):
        GaussianDiffusion(schedule_type="invalid_type")

def test_gaussian_diffusion_valid_schedule_types():
    """Test that GaussianDiffusion accepts valid schedule types"""
    # This test verifies that valid types do not raise exceptions.
    try:
        GaussianDiffusion(schedule_type="linear")
        GaussianDiffusion(schedule_type="cosine")
    except Exception as e:
        pytest.fail(f"Valid schedule types raised exception: {e}")
