#!/bin/bash
set -e

rm -rf build

# Create build directory
mkdir -p build
cd build

# Run CMake
cmake ..

# Build
make -j$(nproc)
