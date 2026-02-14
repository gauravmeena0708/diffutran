#!/bin/bash

# Configuration
CHECKPOINT_PATH="checkpoints_adult_large/checkpoint_epoch_50.pt"
PREPROCESSOR_PATH="checkpoints_adult_large/preprocessor.json"
CONDITION_ENCODER_PATH="checkpoints_adult_large/condition_encoder.joblib"
TEST_DATA_PATH="datasets/adult/test.csv"
OUTPUT_DIR="results_adult_large_epoch50"

echo "Evaluating Checkpoint: $CHECKPOINT_PATH..."

python3 generate_test.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --preprocessor_path "$PREPROCESSOR_PATH" \
  --condition_encoder_path "$CONDITION_ENCODER_PATH" \
  --real_data_path "$TEST_DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples 2000 \
  --condition_col income \
  --sample_condition_dist \
  --hidden_dim 512 \
  --num_layers 6 \
  --num_heads 8

echo "Evaluation complete! Check results in $OUTPUT_DIR"
