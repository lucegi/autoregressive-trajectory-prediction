#!/usr/bin/env bash

set -e  # Exit if anything fails

### CONFIGURATION ###
DATASET_NAME="motion-forecasting"  # Options: sensor, lidar, motion_forecasting, tbv
S5CMD_VERSION="2.0.0"
INSTALL_DIR="$HOME/.local/bin"
#####################

# Resolve the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create target dir as "dataset/" inside script's folder
TARGET_DIR="${SCRIPT_DIR}/dataset"

echo "Installing s5cmd into $INSTALL_DIR..."

# Create install directory if needed
mkdir -p "$INSTALL_DIR"

# Add to PATH for current session
export PATH="$INSTALL_DIR:$PATH"

# Build download URL
S5CMD_OS=$(uname | sed 's/Darwin/macOS/g')
S5CMD_URI="https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_${S5CMD_OS}-64bit.tar.gz"

# Download and extract s5cmd binary
curl -sL "$S5CMD_URI" | tar -C "$INSTALL_DIR" -xvzf - s5cmd

# Check if s5cmd works
if ! command -v s5cmd &>/dev/null; then
    echo "Error: s5cmd is not available in PATH"
    exit 1
fi

echo "✅ s5cmd installed successfully."

# Make sure target dataset directory exists
mkdir -p "$TARGET_DIR"

echo "Downloading Argoverse2 dataset: $DATASET_NAME"
echo "   → Target: $TARGET_DIR"

# Download using anonymous access
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" $TARGET_DIR

echo "Download complete"