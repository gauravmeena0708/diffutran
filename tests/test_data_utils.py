import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.common.data_utils import DataPreprocessor

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numerical_1': np.random.randn(100),
        'numerical_2': np.random.randint(0, 100, 100),
        'categorical_1': ['A', 'B'] * 50,
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], 100)
    })

def test_data_preprocessor_round_trip(sample_dataframe):
    """
    Tests if the DataPreprocessor can fit_transform and then inverse_transform
    the data back to a state close to the original.
    """
    df_original = sample_dataframe.copy()
    
    # Identify column types
    categorical_cols = ['categorical_1', 'categorical_2']
    numerical_cols = ['numerical_1', 'numerical_2']
    
    preprocessor = DataPreprocessor()
    
    # 1. Fit and transform the data
    transformed_data = preprocessor.fit_transform(df_original, categorical_cols=categorical_cols)
    
    assert transformed_data is not None
    assert transformed_data.shape[0] == len(df_original)
    
    # 2. Inverse transform the data
    df_reconstructed = preprocessor.inverse_transform(transformed_data)
    
    # 3. Verify the reconstructed data
    assert df_reconstructed.shape == df_original.shape
    
    # Check if numerical columns are approximately equal
    for col in numerical_cols:
        assert np.allclose(df_original[col], df_reconstructed[col], atol=1e-5)
        
    # Check if categorical columns are identical
    for col in categorical_cols:
        assert (df_original[col] == df_reconstructed[col]).all()

def test_preprocessor_save_load(sample_dataframe, tmp_path):
    """
    Tests if the preprocessor can be saved and loaded correctly.
    """
    preprocessor = DataPreprocessor()
    categorical_cols = ['categorical_1', 'categorical_2']
    
    # Fit on data
    preprocessor.fit_transform(sample_dataframe, categorical_cols=categorical_cols)
    
    # Save
    save_path = tmp_path / "preprocessor.json"
    preprocessor.save(save_path)
    
    # Load into a new instance
    new_preprocessor = DataPreprocessor()
    new_preprocessor.load(save_path)
    
    # Verify state
    assert new_preprocessor.numerical_cols == preprocessor.numerical_cols
    assert new_preprocessor.categorical_cols == preprocessor.categorical_cols
    assert new_preprocessor.categorical_cardinalities == preprocessor.categorical_cardinalities
    assert new_preprocessor.column_info == preprocessor.column_info
    
    # Verify functionality by transforming data and comparing results
    transformed1 = preprocessor.transform(sample_dataframe)
    transformed2 = new_preprocessor.transform(sample_dataframe)
    
    assert np.allclose(transformed1, transformed2)
