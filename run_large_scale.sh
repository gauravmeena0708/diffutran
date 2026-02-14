#!/bin/bash

# Configuration
DATA_PATH="datasets/adult/train.csv"
TEST_DATA_PATH="datasets/adult/test.csv"
CHECKPOINT_DIR="checkpoints_adult_large"
OUTPUT_DIR="results_adult_large"

echo "Starting Large Scale Training..."
python3 train.py \
  --data_path $DATA_PATH \
  --categorical_cols workclass education marital-status occupation relationship race sex native-country \
  --condition_col income \
  --hidden_dim 512 \
  --num_layers 6 \
  --num_heads 8 \
  --num_epochs 100 \
  --batch_size 512 \
  --learning_rate 0.0001 \
  --p_uncond 0.1 \
  --save_dir $CHECKPOINT_DIR

echo "Training complete. Starting Generation and Evaluation..."

python3 generate_test.py \
  --checkpoint_path "$CHECKPOINT_DIR/checkpoint_epoch_final.pt" \
  --preprocessor_path "$CHECKPOINT_DIR/preprocessor.json" \
  --condition_encoder_path "$CHECKPOINT_DIR/condition_encoder.joblib" \
  --real_data_path $TEST_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --num_samples 2000 \
  --condition_col income \
  --sample_condition_dist \
  --hidden_dim 512 \
  --num_layers 6 \
  --num_heads 8

echo "Done! Check results in $OUTPUT_DIR"
