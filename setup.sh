#!/bin/bash

# Before running any of the Carla CLI commands, you need to set up the workspace.
# This script will set up the workspace for you.
# Always run this script from the root of the Carla repository.
# Run this script if you would life to refresh your workspace or initialize it for the first time.
# This script will:
# 1. Clear the logs.
# 2. Update the DVC components.
# 3. Install required Python packages.
# 4. Update the git submodules.
# 6. Build the `algorithms` library.
# 7. Have services up and running using docker compose.

OS=$(uname)
echo "You are running on $OS."

# Clear logs if enabled
echo "Clearing logs..."
if [ -z "$CLEAR_LOGS" ]; then
    echo "-- CLEAR_LOGS is not set. Please set it to true in your environment file if you would like to clear the logs. Skipping log clearing..."
else
    if [ "$CLEAR_LOGS" = "true" ]; then
        (
            cd ./logs
            rm -rf *
        )
        echo "-- Logs cleared."
    else
        echo "-- CLEAR_LOGS is set to false. Skipping log clearing..."
    fi
fi

# Update DVC components
echo "Updating DVC components..."
if [ -z "$DVC_DATA_PATH" ]; then
    echo "-- DVC_DATA_PATH is not set. Please set it to the path of your DVC data in your environment file. Skipping DVC setup..."
else
    if [ "$UPDATE_DVC_CONFIG" = "true" ]; then
        dvc remote modify local url "$DVC_DATA_PATH"
        echo "-- DVC config updated for $OS with remote URL: $DVC_DATA_PATH"
    else
        echo "-- UPDATE_DVC_CONFIG is set to false. Skipping DVC config update..."
    fi
fi

# Install required Python packages
echo "Installing required Python packages..."
poetry install

# Update git submodules
echo "Initializing/Updating git submodules..."
git submodule update --init --recursive

# Setup `algorithms` library
echo "Setting up \`algorithms\` library..."
mkdir algorithmslib
(
    echo "-- Building \`algorithms\` library with default settings [you may customize by building it manually]..."
    cd algorithmslib
    cmake ../third_party/algorithms
    make
)

# Have services up and running using docker compose
echo "Starting services using docker compose..."
docker compose down
if [ $OS = "Darwin" ]; then
    docker compose up -d --build # Use this for MacOS
else
    docker compose --profile linux up -d --build # Use this for Linux
fi
docker system prune -f