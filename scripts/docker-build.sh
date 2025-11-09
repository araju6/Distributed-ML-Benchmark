#!/bin/bash
# Docker build and push script for ML Benchmark container
#
# Usage:
#   ./scripts/docker-build.sh [tag] [--push]
#
# Environment variables:
#   DOCKERHUB_USERNAME - Your Docker Hub username (required for --push)
#
# Examples:
#   ./scripts/docker-build.sh                    # Build with tag 'latest'
#   ./scripts/docker-build.sh v1.0               # Build with tag 'v1.0'
#   ./scripts/docker-build.sh latest --push      # Build and push to Docker Hub
#   DOCKERHUB_USERNAME=myuser ./scripts/docker-build.sh v1.0 --push

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse arguments
TAG="${1:-latest}"
PUSH=false
if [ "$2" = "--push" ] || [ "$1" = "--push" ]; then
    PUSH=true
    if [ "$1" = "--push" ]; then
        TAG="latest"
    fi
fi

# Image naming
IMAGE_NAME="ml-benchmark"
LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"

# Docker Hub image (default to 5ifty6ix56, can be overridden)
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-5ifty6ix56}"
DOCKERHUB_IMAGE="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "======================================================================"
echo "Building Docker image: $LOCAL_IMAGE"
echo "Docker Hub image: $DOCKERHUB_IMAGE"
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

# Check Docker Hub authentication if pushing
if [ "$PUSH" = true ]; then
    echo ""
    echo "Checking Docker Hub authentication..."
    if ! docker info 2>/dev/null | grep -q "Username"; then
        echo "⚠ Warning: Not logged into Docker Hub"
        echo "  Run: docker login"
        echo "  Or set DOCKER_PASSWORD and use: echo \$DOCKER_PASSWORD | docker login --username \$DOCKERHUB_USERNAME --password-stdin"
    else
        echo "✓ Docker Hub authentication detected"
    fi
fi

echo ""
echo "Building image for linux/amd64 (required for P100 GPUs)..."
# Use buildx for cross-platform builds, or regular build if buildx not available
USE_BUILDX=false
if docker buildx version &>/dev/null; then
    USE_BUILDX=true
    # Check if builder exists, create if not
    if ! docker buildx ls | grep -q "ml-benchmark-builder"; then
        echo "Creating buildx builder for cross-platform builds..."
        docker buildx create --name ml-benchmark-builder --use 2>/dev/null || true
    fi
fi

BUILD_SUCCESS=false
if [ "$PUSH" = true ] && [ "$USE_BUILDX" = true ]; then
    # Build and push in one step with buildx
    docker buildx build \
        --platform linux/amd64 \
        --tag "$DOCKERHUB_IMAGE" \
        --push \
        -f Dockerfile \
        .
    BUILD_SUCCESS=$?
elif [ "$USE_BUILDX" = true ]; then
    # Build with buildx and load into local Docker
    docker buildx build \
        --platform linux/amd64 \
        --tag "$LOCAL_IMAGE" \
        --tag "$DOCKERHUB_IMAGE" \
        --load \
        -f Dockerfile \
        .
    BUILD_SUCCESS=$?
else
    # Fallback to regular docker build
    echo "⚠ Warning: docker buildx not available, building for host platform"
    echo "  For P100 GPUs (x86_64), install buildx: docker buildx install"
    docker build \
        --platform linux/amd64 \
        -t "$LOCAL_IMAGE" \
        -f Dockerfile \
        .
    BUILD_SUCCESS=$?
    if [ $BUILD_SUCCESS -eq 0 ]; then
        # Tag for Docker Hub
        docker tag "$LOCAL_IMAGE" "$DOCKERHUB_IMAGE"
    fi
fi

if [ $BUILD_SUCCESS -ne 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✗ Build failed!"
    echo "======================================================================"
    exit 1
fi

# Tag for Docker Hub (if not already tagged by buildx and not pushing)
if [ "$PUSH" != true ] && ! docker images | grep -q "$DOCKERHUB_USERNAME/$IMAGE_NAME.*$TAG"; then
    echo ""
    echo "Tagging image for Docker Hub..."
    docker tag "$LOCAL_IMAGE" "$DOCKERHUB_IMAGE"
    echo "✓ Tagged as: $DOCKERHUB_IMAGE"
fi

# Push to Docker Hub if requested (and not already pushed by buildx)
if [ "$PUSH" = true ] && [ "$USE_BUILDX" != true ]; then
    echo ""
    echo "Pushing to Docker Hub..."
    docker push "$DOCKERHUB_IMAGE"
    PUSH_SUCCESS=$?
    
    if [ $PUSH_SUCCESS -eq 0 ]; then
        echo ""
        echo "======================================================================"
        echo "✓ Push successful!"
        echo "======================================================================"
        echo ""
        echo "Image available at: docker.io/$DOCKERHUB_IMAGE"
        echo ""
        echo "To use in Kubernetes, the image is already configured in k8s/raycluster.yaml"
    else
        echo ""
        echo "======================================================================"
        echo "✗ Push failed!"
        echo "======================================================================"
        exit 1
    fi
elif [ "$PUSH" = true ] && [ "$USE_BUILDX" = true ]; then
    # Push was done by buildx, just show success message
    if [ $BUILD_SUCCESS -eq 0 ]; then
        echo ""
        echo "======================================================================"
        echo "✓ Build and push successful!"
        echo "======================================================================"
        echo ""
        echo "Image available at: docker.io/$DOCKERHUB_IMAGE"
        echo ""
        echo "To use in Kubernetes, the image is already configured in k8s/raycluster.yaml"
    fi
else
    echo ""
    echo "======================================================================"
    echo "✓ Build successful!"
    echo "======================================================================"
    echo ""
    echo "Local image: $LOCAL_IMAGE"
    echo "Docker Hub image (tagged, not pushed): $DOCKERHUB_IMAGE"
    echo ""
    echo "To push to Docker Hub:"
    echo "  docker push $DOCKERHUB_IMAGE"
    echo "  Or: ./scripts/docker-build.sh $TAG --push"
    echo ""
    echo "To run the container:"
    echo "  docker run --gpus all -v \$(pwd)/config.yaml:/workspace/config.yaml:ro -v \$(pwd)/results:/workspace/results $LOCAL_IMAGE"
    echo ""
    echo "To test with docker-compose:"
    echo "  docker-compose -f docker-compose.ray.yml up"
    echo ""
fi

