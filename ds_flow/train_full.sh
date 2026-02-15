#!/bin/bash

# Configuration
DATA_PATH="datasets/adult/train.csv"
SAVE_DIR="checkpoints_dsflow_full"
CATEGORICAL_COLS="workclass education marital-status occupation relationship race sex native-country income"

# Hyperparameters (Matched to Model 1 where applicable)
EPOCHS=100
BATCH_SIZE=512
D_MODEL=512
LAYERS=6
HEADS=8

echo "Starting DS-Flow Full Training ($EPOCHS Epochs)..."
python3 ds_flow/train.py \
  --data_path $DATA_PATH \
  --save_dir $SAVE_DIR \
  --categorical_cols $CATEGORICAL_COLS \
  --condition_col income \
  --num_epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --d_model $D_MODEL \
  --n_layers $LAYERS \
  --n_heads $HEADS

echo "Training Complete! Checkpoints in $SAVE_DIR"
