#!/bin/bash

# Before running any of the Carla CLI commands, you need to set up the workspace.
# This script will set up the workspace for you.
# Always run this script from the root of the Carla repository.
# Run this script if you would life to refresh your workspace or initialize it for the first time.
# This script will:
# 1. Update the DVC components.
# 2. Install required Python packages.
# 3. Install the Argoverse dependencies.
# 4. Update the git submodules.
# 5. Build the `algorithms` library.

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

# Install required Python packages
echo "Installing required Python packages..."
pip install -r requirements.txt
if [ -z "$INSTALL_SIMPAN_DEPENDENCIES" ]; then
    echo "INSTALL_SIMPAN_DEPENDENCIES is not set. Please set it to true in your environment file if you would like to install the dependencies for the SimPan project. Skipping SimPan dependencies installation..."
else
    if [ "$INSTALL_SIMPAN_DEPENDENCIES" = "true" ]; then
        echo "Installing SimPan dependencies..."
        pip install -r requirements-simpan.txt
    else
        echo "INSTALL_SIMPAN_DEPENDENCIES is set to false. Skipping SimPan dependencies installation..."
    fi
fi

# Install agro dependencies - https://argoverse.github.io/user-guide/argoverse_2.html
echo "Installing Argoverse dependencies..."
if [ -z "$AV2_DIRECTORY" ]; then
    echo "AV2_DIRECTORY is not set. Please set it to true in your environment file if you would like to install the dependencies for the SimPan project. Skipping Argoverse dependencies installation..."
else
    if [ ! -d "$AV2_DIRECTORY" ]; then
        echo "AV2_DIRECTORY does not exist. Creating directory..."
        mkdir -p $AV2_DIRECTORY
    fi
    (
        cd $AV2_DIRECTORY
        if [ ! -d "av2-api" ]; then
            echo "Cloning Argoverse API repository..."
            git clone git@github.com:argoverse/av2-api.git
        else
            echo "Argoverse API repository already exists. Pulling latest changes..."
            (
                cd av2-api
                git pull
            )
        fi
        cd av2-api && cargo update
    )
    pip install -e $AV2_DIRECTORY/av2-api
fi

# Update git submodules
echo "Initializing/Updating git submodules..."
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