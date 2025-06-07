#!/bin/bash

task="$1" # Task name of training dataset
inp="$2" # Input path
out="$3" # Output path

# Abort on error
set -e

# Export environment variables
export base="$BASE_PATH"
export nnUNet_raw="$BASE_PATH/data/processed/segmentation/nnunet/training/tasks"
export nnUNet_preprocessed="$BASE_PATH/data/processed/segmentation/nnunet/training/nnUNet_preprocessed"
export nnUNet_results="$BASE_PATH/data/models"
export num_processes_preprocessing=1
export num_processes_preprocessing=1

# Get the task number
digits="${task:7:3}"

# Train nnUNet
echo "Predicting ..."
nnUNetv2_predict -i $inp -o $out -d "$digits" -c 3d_fullres -f all