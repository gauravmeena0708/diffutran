import pytest
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
