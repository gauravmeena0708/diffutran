import torch
import pytest
import numpy as np
import pandas as pd
from src.diffutron.diffusion import GaussianDiffusion
from src.flow_model.models.flow_matching import OTCFM
from src.flow_model.data.tokenizer import QuantileTokenizer
from src.flow_model.models.dual_stream import DualStreamTransformer

def test_ddim_sample_shape():
    num_timesteps = 100
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps)
    
    class MockModel(torch.nn.Module):
        def forward(self, x, t, condition=None):
            return x # predict noise = x for simplicity
            
    model = MockModel()
    shape = (10, 5)
    
    # Test DDIM (eta=0.0) with striding
    samples = diffusion.sample(model, shape, num_inference_steps=10, eta=0.0, device='cpu')
    assert samples.shape == shape

def test_dpm_solver_shape():
    class DummyConfig:
        d_model = 16
        sigma_min = 1e-4
    
    class DummyTokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.numerical_cols = ['n1']
            self.categorical_cols = []
            self.num_projections = torch.nn.ModuleDict({'col_n1': torch.nn.Linear(1, 16)})
        def forward(self, x_num, x_cat):
            return torch.randn(x_num.shape[0], 1, 16)
        def inverse_transform(self, x):
            return pd.DataFrame({'n1': np.zeros(x.shape[0])})
        def _module_key(self, col): return f"col_{col}"

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x, t, y=None):
            return torch.zeros_like(x)
            
    engine = OTCFM(DummyBackbone(), DummyTokenizer(), DummyConfig())
    
    # Test Euler
    samples_euler = engine.sample(n_samples=5, steps=5, solver='euler')
    assert samples_euler.shape == (5, 1, 16)
    
    # Test DPM++ (Midpoint)
    samples_dpm = engine.sample(n_samples=5, steps=5, solver='dpm++')
    assert samples_dpm.shape == (5, 1, 16)

def test_temporal_tokenizer():
    df = pd.DataFrame({
        'n1': np.random.randn(10),
        'c1': ['A', 'B'] * 5
    })
    # is_temporal=True, sequence_len=2. Total 5 sequences.
    tokenizer = QuantileTokenizer(d_model=8, is_temporal=True, sequence_len=2, n_quantiles=5)
    tokenizer.fit(df)
    
    x_num, x_cat = tokenizer.preprocess(df)
    assert x_num.shape == (5, 2, 1) # [B, T, N]
    assert x_cat.shape == (5, 2, 1)
    
    tokens = tokenizer.forward(x_num, x_cat)
    assert tokens.shape == (5, 2, 2, 8) # [B, T, N, D] (1 num + 1 cat)
    
    df_inv = tokenizer.inverse_transform(tokens)
    assert len(df_inv) == 10
    assert 'n1' in df_inv.columns

def test_temporal_transformer():
    model = DualStreamTransformer(d_model=16, n_layers=1, n_heads=2, d_ff=32)
    # [B, T, N, D]
    x = torch.randn(2, 5, 3, 16)
    t = torch.rand(2)
    out = model(x, t)
    assert out.shape == (2, 5, 3, 16)
