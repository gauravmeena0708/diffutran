"""
Configuration file for Diffusion Transformer training
Modify these parameters to experiment with different settings
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_CONFIG = {
    'data_path': 'your_data.csv',
    'categorical_cols': ['category_1', 'category_2'],  # List of categorical column names
    'condition_cols': None,  # None for unconditional, or ['col1', 'col2'] for conditional
    'val_split': 0.1,  # Validation split ratio
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
MODEL_CONFIG = {
    # Transformer architecture
    'hidden_dim': 256,      # Hidden dimension (64, 128, 256, 512)
    'num_layers': 6,        # Number of transformer layers (2-8)
    'num_heads': 8,         # Number of attention heads (4, 8, 16)
    'mlp_ratio': 4.0,       # MLP expansion ratio
    'dropout': 0.1,         # Dropout rate (0.0-0.3)
    
    # Diffusion process
    'num_timesteps': 1000,  # Number of diffusion steps (500-1000)
    'beta_start': 1e-4,     # Starting beta value
    'beta_end': 0.02,       # Ending beta value
    'schedule_type': 'linear',  # 'linear' or 'cosine'
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    'batch_size': 128,      # Batch size (32, 64, 128, 256, 512)
    'num_epochs': 100,      # Number of training epochs
    'learning_rate': 1e-4,  # Learning rate (5e-5, 1e-4, 2e-4)
    'save_interval': 10,    # Save checkpoint every N epochs
    'device': 'cuda',       # 'cuda' or 'cpu'
    'save_dir': 'checkpoints',  # Directory to save models
}

# ============================================================================
# GENERATION CONFIGURATION
# ============================================================================
GENERATION_CONFIG = {
    'num_samples': 5000,    # Number of synthetic samples to generate
    'batch_size': 256,      # Generation batch size
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVALUATION_CONFIG = {
    'target_column': 'target',  # Column to use for ML efficacy test
    'task_type': 'classification',  # 'classification' or 'regression'
    'save_dir': 'evaluation_results',  # Directory to save evaluation results
}

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

# Small dataset (< 1,000 samples)
SMALL_DATASET_CONFIG = {
    'model': {
        'hidden_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'num_timesteps': 500,
    },
    'training': {
        'batch_size': 32,
        'num_epochs': 200,
        'learning_rate': 1e-4,
    }
}

# Medium dataset (1,000-10,000 samples)
MEDIUM_DATASET_CONFIG = {
    'model': {
        'hidden_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'num_timesteps': 1000,
    },
    'training': {
        'batch_size': 128,
        'num_epochs': 100,
        'learning_rate': 1e-4,
    }
}

# Large dataset (> 10,000 samples)
LARGE_DATASET_CONFIG = {
    'model': {
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 16,
        'num_timesteps': 1000,
    },
    'training': {
        'batch_size': 256,
        'num_epochs': 50,
        'learning_rate': 1e-4,
    }
}

# High quality (slower but better results)
HIGH_QUALITY_CONFIG = {
    'model': {
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 16,
        'num_timesteps': 1000,
        'dropout': 0.1,
    },
    'training': {
        'batch_size': 64,
        'num_epochs': 200,
        'learning_rate': 5e-5,
    }
}

# Fast training (quicker but lower quality)
FAST_CONFIG = {
    'model': {
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 4,
        'num_timesteps': 500,
    },
    'training': {
        'batch_size': 256,
        'num_epochs': 30,
        'learning_rate': 2e-4,
    }
}


def get_config(preset=None):
    """
    Get configuration with optional preset
    
    Args:
        preset: One of ['small', 'medium', 'large', 'high_quality', 'fast', None]
    
    Returns:
        Dictionary with all configurations
    """
    config = {
        'data': DATA_CONFIG.copy(),
        'model': MODEL_CONFIG.copy(),
        'training': TRAINING_CONFIG.copy(),
        'generation': GENERATION_CONFIG.copy(),
        'evaluation': EVALUATION_CONFIG.copy(),
    }
    
    # Apply preset if specified
    if preset == 'small':
        config['model'].update(SMALL_DATASET_CONFIG['model'])
        config['training'].update(SMALL_DATASET_CONFIG['training'])
    elif preset == 'medium':
        config['model'].update(MEDIUM_DATASET_CONFIG['model'])
        config['training'].update(MEDIUM_DATASET_CONFIG['training'])
    elif preset == 'large':
        config['model'].update(LARGE_DATASET_CONFIG['model'])
        config['training'].update(LARGE_DATASET_CONFIG['training'])
    elif preset == 'high_quality':
        config['model'].update(HIGH_QUALITY_CONFIG['model'])
        config['training'].update(HIGH_QUALITY_CONFIG['training'])
    elif preset == 'fast':
        config['model'].update(FAST_CONFIG['model'])
        config['training'].update(FAST_CONFIG['training'])
    
    return config


if __name__ == '__main__':
    # Example: Print configuration
    import json
    
    print("Default Configuration:")
    print(json.dumps(get_config(), indent=2))
    
    print("\n\nHigh Quality Preset:")
    print(json.dumps(get_config('high_quality'), indent=2))
