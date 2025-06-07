#!/bin/bash

task="$1" # Task name of training dataset

# Abort on error
set -e

# Export environment variables
export base="$BASE_PATH"
export nnUNet_raw="$BASE_PATH/data/processed/segmentation/nnunet/training/tasks"
export nnUNet_preprocessed="$BASE_PATH/data/processed/segmentation/nnunet/training/nnUNet_preprocessed"
export nnUNet_results="$BASE_PATH/data/models"

# Get the task number
digits="${task:7:3}"

# Train nnUNet
echo "Training nnUNet for task $task ..."
echo "Preprocessing ..."
nnUNetv2_plan_and_preprocess -d "$digits" -c 3d_fullres

echo "Training ..."
nnUNetv2_train "$task" 3d_fullres all
