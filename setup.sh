#!/bin/bash

# Before running any of the Carla CLI commands, you need to set up the workspace.
# This script will set up the workspace for you.

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