import pytest
import sys
from unittest.mock import MagicMock

# Import the code under test
# This relies on conftest.py mocking torch if it's missing
from models.embeddings import FeatureEmbedder, SinusoidalPositionalEmbedding


# Helper to check if torch is mocked
def is_torch_mocked():
    try:
        import torch

        return (
            isinstance(torch, MagicMock)
            or hasattr(torch, "MagicMock")
            or "mock" in str(type(torch)).lower()
        )
    except ImportError:
        return True


class TestFeatureEmbedder:

    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        if is_torch_mocked():
            import torch

            if hasattr(torch, "cat") and isinstance(torch.cat, MagicMock):
                torch.cat.reset_mock()

    @pytest.fixture
    def setup_embedder(self):
        def _setup(num_numerical, categorical_cardinalities, d_model, batch_size):
            embedder = FeatureEmbedder(
                num_numerical, categorical_cardinalities, d_model
            )

            # Create mocks for inputs
            if is_torch_mocked():
                x_num = MagicMock()
                # Mock shape and slicing
                x_num.shape = (batch_size, num_numerical)
                # Slicing returns another mock which needs to support .unsqueeze(1)
                # x_num[:, i] -> __getitem__
                x_num_slice = MagicMock()
                x_num.__getitem__.return_value = x_num_slice
                x_num_slice.unsqueeze.return_value = MagicMock()

                x_cat = MagicMock()
                x_cat.shape = (batch_size, len(categorical_cardinalities))
                x_cat_slice = MagicMock()
                x_cat.__getitem__.return_value = x_cat_slice
                # x_cat[:, i].long()
                x_cat_slice.long.return_value = MagicMock()
            else:
                import torch

                x_num = torch.randn(batch_size, num_numerical)
                x_cat = torch.randint(
                    0, 3, (batch_size, len(categorical_cardinalities))
                )

            return embedder, x_num, x_cat

        return _setup

    def test_forward_mixed_features(self, setup_embedder):
        num_numerical = 2
        categorical_cardinalities = [3, 4]
        d_model = 16
        batch_size = 5

        embedder, x_num, x_cat = setup_embedder(
            num_numerical, categorical_cardinalities, d_model, batch_size
        )

        out = embedder(x_num, x_cat)

        # Verify output
        if is_torch_mocked():
            import torch

            # Verify torch.cat called
            torch.cat.assert_called()
            args, kwargs = torch.cat.call_args
            embeddings_list = args[0]
            # Check length of list passed to cat
            assert len(embeddings_list) == num_numerical + len(
                categorical_cardinalities
            )
            # Check dim argument
            assert kwargs.get("dim") == 1 or args[1] == 1
        else:
            assert out.shape == (
                batch_size,
                num_numerical + len(categorical_cardinalities),
                d_model,
            )

    def test_forward_numerical_only(self, setup_embedder):
        num_numerical = 2
        categorical_cardinalities = []
        d_model = 16
        batch_size = 5

        embedder, x_num, x_cat = setup_embedder(
            num_numerical, categorical_cardinalities, d_model, batch_size
        )

        out = embedder(x_num, None)

        if is_torch_mocked():
            import torch

            torch.cat.assert_called()
            args, _ = torch.cat.call_args
            embeddings_list = args[0]
            assert len(embeddings_list) == num_numerical
        else:
            assert out.shape == (batch_size, num_numerical, d_model)

    def test_forward_categorical_only(self, setup_embedder):
        num_numerical = 0
        categorical_cardinalities = [3, 4]
        d_model = 16
        batch_size = 5

        embedder, x_num, x_cat = setup_embedder(
            num_numerical, categorical_cardinalities, d_model, batch_size
        )

        out = embedder(None, x_cat)

        if is_torch_mocked():
            import torch

            torch.cat.assert_called()
            args, _ = torch.cat.call_args
            embeddings_list = args[0]
            assert len(embeddings_list) == len(categorical_cardinalities)
        else:
            assert out.shape == (batch_size, len(categorical_cardinalities), d_model)


class TestSinusoidalPositionalEmbedding:
    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        if is_torch_mocked():
            import torch

            if hasattr(torch, "cat") and isinstance(torch.cat, MagicMock):
                torch.cat.reset_mock()
            if hasattr(torch, "sin") and isinstance(torch.sin, MagicMock):
                torch.sin.reset_mock()

    def test_forward(self):
        dim = 32
        embedder = SinusoidalPositionalEmbedding(dim)

        if is_torch_mocked():
            timesteps = MagicMock()
            timesteps.device = "cpu"
            # timesteps[:, None] -> __getitem__
            timesteps.__getitem__.return_value = MagicMock()

            out = embedder(timesteps)

            import torch

            torch.cat.assert_called()
            torch.sin.assert_called()
            torch.cos.assert_called()
        else:
            import torch

            timesteps = torch.randint(0, 1000, (10,))
            out = embedder(timesteps)
            assert out.shape == (10, dim)
