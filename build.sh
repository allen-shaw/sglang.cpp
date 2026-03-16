#!/bin/bash
set -e

rm -rf build

# Create build directory
mkdir -p build
cd build

# Run CMake (adjust TORCH_PATH if your LibTorch is elsewhere)
TORCH_PATH="${TORCH_PATH:-$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null || echo '')}"
cmake .. -DCMAKE_PREFIX_PATH="$TORCH_PATH"

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure
