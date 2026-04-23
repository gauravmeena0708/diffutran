import pandas as pd
import numpy as np
import os

def generate_synthetic_temporal_data(num_sequences=100, seq_len=10, save_path='workdir/data/processed/synthetic_temporal/train.csv'):
    """
    Generates a synthetic temporal dataset:
    - 3 numerical features (sine waves with noise)
    - 2 categorical features (state transitions)
    - 1 target column
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = []
    for seq_id in range(num_sequences):
        # Base trend for the sequence
        base_val = np.random.rand()
        category_state = np.random.choice(['High', 'Low', 'Medium'])
        
        for t in range(seq_len):
            n1 = base_val + np.sin(t * 0.5) + np.random.normal(0, 0.1)
            n2 = base_val * 2 + np.cos(t * 0.5) + np.random.normal(0, 0.1)
            n3 = np.random.normal(0, 1)
            
            # Categorical state that evolves slightly
            c1 = category_state
            c2 = 'Work' if t < 5 else 'Rest'
            
            # Target depends on sequence trends
            target = 1 if (n1 + n2) > 1.5 else 0
            
            data.append({
                'seq_id': seq_id,
                'time_step': t,
                'feature_1': n1,
                'feature_2': n2,
                'feature_3': n3,
                'state': c1,
                'activity': c2,
                'target': target
            })
            
    df = pd.DataFrame(data)
    # Ensure seq_id and time_step are handled as context or dropped if we want pure tabular sequence
    # For our DS-Flow testing, we just need [B*T, F] in order.
    df.to_csv(save_path, index=False)
    print(f"Generated synthetic temporal data at {save_path}")

if __name__ == '__main__':
    generate_synthetic_temporal_data()
