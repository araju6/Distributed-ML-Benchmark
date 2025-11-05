#!/bin/bash
# Docker build script for ML Benchmark container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

IMAGE_NAME="ml-benchmark"
TAG="${1:-latest}"

echo "======================================================================"
echo "Building Docker image: $IMAGE_NAME:$TAG"
echo "======================================================================"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if nvidia-docker runtime is available (for GPU support)
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "✓ NVIDIA Docker runtime detected"
else
    echo "⚠ Warning: NVIDIA Docker runtime not detected. GPU support may not work."
    echo "  Install nvidia-container-toolkit for GPU support"
fi

echo ""
echo "Building image..."
docker build \
    -t "$IMAGE_NAME:$TAG" \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Build successful!"
    echo "======================================================================"
    echo ""
    echo "Image: $IMAGE_NAME:$TAG"
    echo ""
    echo "To run the container:"
    echo "  docker run --gpus all -v \$(pwd)/config.yaml:/workspace/config.yaml:ro -v \$(pwd)/results:/workspace/results $IMAGE_NAME:$TAG"
    echo ""
    echo "To test with docker-compose:"
    echo "  docker-compose -f docker-compose.ray.yml up"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "✗ Build failed!"
    echo "======================================================================"
    exit 1
fi

