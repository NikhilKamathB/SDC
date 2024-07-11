#!/bin/bash

# Before running any of the Carla CLI commands, you need to set up the workspace.
# This script will set up the workspace for you.
# Always run this script from the root of the Carla repository.
# Run this script if you would life to refresh your workspace or initialize it for the first time.
# This script will:
# 1. Update the DVC components.
# 2. Update the git submodules.
# 3. Build the `algorithms` library.

OS=$(uname)
echo "You are running on $OS."

# Update DVC components
echo "Updating DVC components..."
if [ -z "$DVC_DATA_PATH" ]; then
    echo "DVC_DATA_PATH is not set. Please set it to the path of your DVC data in your environment file. Skipping DVC setup..."
else
    dvc remote modify local url "$DVC_DATA_PATH"
    echo "DVC config updated for $OS with remote URL: $DVC_DATA_PATH"
fi

# Update git submodules
echo "Updating git submodules..."
git submodule update --init --recursive

# Setup `algorithms` library
echo "Setting up \`algorithms\` library..."
mkdir algorithmslib
(
    echo "Building \`algorithms\` library with default settings [you may customize by building it manually]..."
    cd algorithmslib
    cmake ../third_party/algorithms
    make
)