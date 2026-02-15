import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.diffutron.transformer import TransformerModule

class TestTransformerModule(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 2
        self.dropout = 0.1
        self.dim_feedforward = 128
        self.batch_size = 4
        self.seq_len = 10

        self.model = TransformerModule(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            dim_feedforward=self.dim_feedforward
        )

    def test_initialization(self):
        """Test if the model is initialized correctly."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model.encoder, nn.TransformerEncoder)
        # Check number of layers
        self.assertEqual(len(self.model.encoder.layers), self.num_layers)

    def test_forward_output_shape(self):
        """Test output shape of forward pass."""
        src = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.model(src)

        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_values_change(self):
        """Test that output is different from input (transformation happened)."""
        src = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.model(src)

        # It's unlikely that output equals input exactly unless initialization is identity (it's not)
        self.assertFalse(torch.allclose(src, output))

    def test_gradients(self):
        """Test that gradients flow through the model."""
        src = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        output = self.model(src)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(src.grad)
        # Check that gradients are not zero (or at least some are not zero)
        self.assertTrue(torch.any(src.grad != 0))

    def test_device_movement(self):
        """Test moving model to device (if cuda available, else skip or just check cpu)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        src = torch.randn(self.batch_size, self.seq_len, self.d_model).to(device)
        output = model(src)
        self.assertEqual(output.device.type, device.type)

if __name__ == '__main__':
    unittest.main()
