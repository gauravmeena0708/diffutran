#!/bin/bash

# Configuration
DATA_PATH="datasets/adult/train.csv"
SAVE_DIR="checkpoints_dsflow_verify"
OUTPUT_DIR="results_dsflow_verify"
CATEGORICAL_COLS="workclass education marital-status occupation relationship race sex native-country income"

echo "Step 1: Training DS-Flow (1 Epoch)..."
python3 ds_flow/train.py \
  --data_path $DATA_PATH \
  --save_dir $SAVE_DIR \
  --categorical_cols $CATEGORICAL_COLS \
  --num_epochs 1 \
  --batch_size 128 \
  --d_model 256

echo "Step 2: Generation..."
python3 ds_flow/sampler.py \
  --checkpoint_path "$SAVE_DIR/model_final.pt" \
  --tokenizer_path "$SAVE_DIR/tokenizer.joblib" \
  --output_dir $OUTPUT_DIR \
  --num_samples 20 \
  --steps 10 \
  --d_model 256

echo "Verification Complete! Check $OUTPUT_DIR/synthetic_data.csv"
