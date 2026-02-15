import sys
from unittest.mock import MagicMock

# Check if torch is available
try:
    import torch
except ImportError:
    # Mock torch
    mock_torch = MagicMock()
    mock_nn = MagicMock()
    mock_torch.nn = mock_nn
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = mock_nn

    # Mock nn.Module and others
    class MockModule:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            if hasattr(self, "forward"):
                return self.forward(*args, **kwargs)
            return MagicMock()

    mock_nn.Module = MockModule
    mock_nn.ModuleList = list
    mock_nn.Linear = MagicMock()  # Returns a MagicMock when called
    mock_nn.Embedding = MagicMock()  # Returns a MagicMock when called

    # Mock specific torch functions used in models/embeddings.py
    mock_torch.cat = MagicMock()
    # Also arange, exp, sin, cos for SinusoidalPositionalEmbedding
    mock_torch.arange = MagicMock()
    mock_torch.exp = MagicMock()
    mock_torch.sin = MagicMock()
    mock_torch.cos = MagicMock()

    mock_torch.randn = MagicMock(return_value=MagicMock())
    mock_torch.randint = MagicMock(return_value=MagicMock())
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
