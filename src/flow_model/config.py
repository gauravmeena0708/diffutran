from dataclasses import dataclass

@dataclass
class DSFlowConfig:
    # Model Architecture
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Flow Matching
    sigma_min: float = 1e-4
    
    # Training
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    
    # Data
    quantile_noise: int = 1e-3 # Noise added before quantile transform to prevent discrete artifacts
