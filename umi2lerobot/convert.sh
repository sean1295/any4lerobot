#!/bin/bash

# --- Configuration ---
# Path to the source UMI Zarr dataset (e.g., dataset.zarr or dataset.zarr.zip)
SRC_DATASET_PATH="/standard/liverobotics/datasets_issac/isaaclab/case_open.zarr.zip"

# Directory where the converted LeRobot dataset will be saved
OUTPUT_DIR="/standard/liverobotics/datasets_issac/isaaclab/lerobot_umi_converted"

# Task description (replace with your actual task)
TASK_DESCRIPTION="Example UMI task description"

# Robot type (optional, replace if known)
ROBOT_TYPE="umi_robot"

# Frames per second (optional, replace with actual FPS if known)
FPS=15

# Hugging Face Hub Repo ID (optional, required if pushing to hub)
# REPO_ID="YourHFUsername/lerobot-umi-dataset"
REPO_ID="" # Leave empty if not pushing

# --- Flags ---
# Set to "--use-videos" to store images as videos (recommended), or "" to store as images
USE_VIDEOS_FLAG="--use-videos"

# Set to "--push-to-hub" to upload to Hugging Face Hub, or "" to save locally only
PUSH_TO_HUB_FLAG=""
# PUSH_TO_HUB_FLAG="--push-to-hub" # Uncomment to enable pushing

# --- Execute Script ---
echo "Starting UMI to LeRobot conversion..."
python umi2lerobot.py \
    --src-path "$SRC_DATASET_PATH" \
    --output-path "$OUTPUT_DIR" \
    --task-desc "$TASK_DESCRIPTION" \
    --robot-type "$ROBOT_TYPE" \
    --fps $FPS \
    ${REPO_ID:+--repo-id "$REPO_ID"} \
    $USE_VIDEOS_FLAG \
    $PUSH_TO_HUB_FLAG

echo "Conversion script finished."
