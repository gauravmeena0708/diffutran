import os
import subprocess
import pandas as pd
import numpy as np
import json
import argparse
import time
import sys
from datetime import datetime

# Add project root to path to allow imports from scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.experiment_utils import calculate_csr_from_constraint_strings

def run_command(cmd, desc):
    print(f"--- {desc} ---")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success!")
    return result

def main(args):
    base_data_dir = "workdir/data/processed"
    base_model_dir = "workdir/models"
    base_syn_dir = "workdir/synthetic"
    base_result_dir = "workdir/results"
    
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_syn_dir, exist_ok=True)
    os.makedirs(base_result_dir, exist_ok=True)
    
    datasets = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
    
    all_results = []
    
    for ds in datasets:
        print(f"\n{'='*20}\nProcessing Dataset: {ds}\n{'='*20}")
        data_path = os.path.join(base_data_dir, ds, "train.csv")
        if not os.path.exists(data_path):
            print(f"Skipping {ds}: train.csv not found.")
            continue
            
        df = pd.read_csv(data_path)
        
        # Profile Dataset
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = df.columns[-1] 
        
        if target_col in cat_cols: cat_cols.remove(target_col)
        if target_col in num_cols: num_cols.remove(target_col)
        
        constraint_col = cat_cols[0] if cat_cols else None
        constraint_val = df[constraint_col].mode()[0] if constraint_col else None
        
        is_temporal = "temporal" in ds.lower()
        seq_len = 10 if is_temporal else 1
        
        # 1. Train DS-Flow
        ds_model_dir = os.path.join(base_model_dir, ds, "dsflow")
        train_cmd = [
            "python", "-m", "src.flow_model.train",
            "--data_path", data_path,
            "--save_dir", ds_model_dir,
            "--num_epochs", "1" if args.debug else str(args.epochs),
            "--batch_size", "256",
            "--d_model", "128" if args.debug else "512"
        ]
        if cat_cols:
            train_cmd.extend(["--categorical_cols"] + cat_cols)
        if target_col:
            train_cmd.extend(["--condition_col", target_col])
        if is_temporal:
            train_cmd.extend(["--is_temporal", "--sequence_len", str(seq_len)])
            
        run_command(train_cmd, f"Training DS-Flow on {ds}")
        
        # 2. Train DiffuTron
        m1_model_dir = os.path.join(base_model_dir, ds, "diffutron")
        if not is_temporal:
            train_m1_cmd = [
                "python", "scripts/train.py",
                "--data_path", data_path,
                "--save_dir", m1_model_dir,
                "--num_epochs", "1" if args.debug else str(args.epochs),
                "--batch_size", "256",
                "--hidden_dim", "128" if args.debug else "256"
            ]
            if cat_cols:
                train_m1_cmd.extend(["--categorical_cols"] + cat_cols)
            if target_col:
                train_m1_cmd.extend(["--condition_col", target_col])
                
            run_command(train_m1_cmd, f"Training DiffuTron on {ds}")
            
        # 3. Sample DS-Flow
        syn_ds_dir = os.path.join(base_syn_dir, ds)
        os.makedirs(syn_ds_dir, exist_ok=True)
        
        n_samp = 100 if args.debug else 1000
        
        start_ds = time.time()
        sample_cmd = [
            "python", "-m", "src.flow_model.sampler",
            "--checkpoint_path", os.path.join(ds_model_dir, "model_final.pt"),
            "--tokenizer_path", os.path.join(ds_model_dir, "tokenizer.pt"),
            "--dataset_name", ds,
            "--output_dir", syn_ds_dir,
            "--num_samples", str(n_samp),
            "--steps", "10",
            "--solver", "dpm++",
            "--d_model", "128" if args.debug else "512"
        ]
        if target_col:
            sample_cmd.extend(["--condition_col", target_col])
        if constraint_col and constraint_val:
            sample_cmd.extend(["--constraint", f"{constraint_col}={constraint_val}"])
            
        run_command(sample_cmd, f"Sampling DS-Flow for {ds}")
        ds_eff = (time.time() - start_ds) / (n_samp / 1000)

        # 4. Sample DiffuTron
        m1_eff = 0
        if not is_temporal:
            start_m1 = time.time()
            sample_m1_cmd = [
                "python", "scripts/generate_test.py",
                "--checkpoint_path", os.path.join(m1_model_dir, "checkpoint_epoch_final.pt"),
                "--preprocessor_path", os.path.join(m1_model_dir, "preprocessor.json"),
                "--condition_encoder_path", os.path.join(m1_model_dir, "condition_encoder.json"),
                "--real_data_path", data_path,
                "--output_dir", syn_ds_dir,
                "--num_samples", str(n_samp),
                "--num_inference_steps", "20",
                "--eta", "0.0",
                "--hidden_dim", "128" if args.debug else "256"
            ]
            if target_col:
                sample_m1_cmd.extend(["--condition_col", target_col])
            if constraint_col and constraint_val:
                sample_m1_cmd.extend(["--constraint", f"{constraint_col}={constraint_val}"])
                
            run_command(sample_m1_cmd, f"Sampling DiffuTron for {ds}")
            m1_eff = (time.time() - start_m1) / (n_samp / 1000)

        # 5. Evaluate
        syn_files = [f for f in os.listdir(syn_ds_dir) if f.endswith(".csv")]
        for sf in syn_files:
            sf_path = os.path.join(syn_ds_dir, sf)
            method = "dsflow" if "flow_model" in sf else "diffutron"
            eff = ds_eff if method == "dsflow" else m1_eff
            
            # Constraints for CSR
            current_constraints = [f"{constraint_col}={constraint_val}"] if constraint_col else []
            
            eval_cmd = [
                "python", "-c", f"""
import pandas as pd
import sys
import os
sys.path.insert(0, os.getcwd())
from src.common.evaluation_metrics import EvaluationFramework
from scripts.experiment_utils import calculate_csr_from_constraint_strings
import json

real = pd.read_csv('{data_path}')
syn = pd.read_csv('{sf_path}')

# CSR
constraints = {current_constraints}
csr_score, _, _ = calculate_csr_from_constraint_strings(syn, constraints)

# Clean temporal
if 'seq_id' in real.columns: real = real.drop(columns=['seq_id', 'time_step'])
if 'seq_id' in syn.columns: syn = syn.drop(columns=['seq_id', 'time_step'])

target = '{target_col}'
evaluator = EvaluationFramework(real, syn, target_col=target)
fid = evaluator.evaluate_fidelity()
ks_avg = sum([v for k, v in fid.items() if 'KS_Stat' in k]) / (len([k for k in fid.items() if 'KS_Stat' in k]) or 1)
util = evaluator.evaluate_utility()
tstr = util.get('TSTR_F1', 0) or util.get('TSTR_R2', 0)

priv = evaluator.evaluate_privacy()
div = evaluator.evaluate_diversity()
cov_avg = sum(div.values()) / (len(div) or 1)

result = {{
    'dataset': '{ds}',
    'method': '{method}',
    'avg_ks': round(ks_avg, 4),
    'tstr': round(tstr, 4),
    'csr': round(csr_score, 2),
    'privacy_mia': round(priv.get('MIA_AUC', 0.5), 4),
    'diversity_cov': round(cov_avg, 4),
    'efficiency_1k_sec': round({eff}, 2)
}}
print(json.dumps(result))
"""
            ]
            res = run_command(eval_cmd, f"Evaluating {sf}")
            if res.returncode == 0:
                try:
                    all_results.append(json.loads(res.stdout.splitlines()[-1]))
                except:
                    pass

    # Save Aggregate Results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(base_result_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nBATCH EXPERIMENT COMPLETE. Results saved to {results_path}")
        print(results_df.to_string())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help="Run with small batch/epochs for testing.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs per dataset.")
    args = parser.parse_args()
    main(args)
