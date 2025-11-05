# Docker Containerization Guide

This guide explains how to build and run the ML Compiler Benchmark Framework using Docker containers.

**Location**: This file is located at `docker/README.md` in the project root.

## Prerequisites

- Docker installed (version 20.10+)
- NVIDIA Docker runtime (nvidia-container-toolkit) for GPU support
- Docker Compose (optional, for multi-container Ray cluster)

### Install NVIDIA Docker Runtime

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Building the Image

### Option 1: Using the build script

```bash
./scripts/docker-build.sh [tag]
```

Example:
```bash
./scripts/docker-build.sh v1.0
```

### Option 2: Using Docker directly

```bash
docker build -t ml-benchmark:latest -f Dockerfile .
```

## Running Containers

### Single Container (Sequential Execution)

Run benchmarks sequentially in a single container:

```bash
docker run --rm --gpus all \
  -v $(pwd)/config.yaml:/workspace/config.yaml:ro \
  -v $(pwd)/results:/workspace/results \
  ml-benchmark:latest \
  python /workspace/run_benchmark.py
```

### Single Container (Ray Distributed)

Run distributed benchmarks with Ray (requires GPU access):

```bash
docker run --rm --gpus all \
  -v $(pwd)/config.yaml:/workspace/config.yaml:ro \
  -v $(pwd)/results:/workspace/results \
  ml-benchmark:latest \
  python /workspace/run_benchmark.py --distributed
```

### Multi-Container Ray Cluster

Use Docker Compose to run a Ray cluster with multiple workers:

```bash
# Start Ray cluster (head + workers)
docker-compose -f docker-compose.ray.yml up -d ray-head ray-worker-1 ray-worker-2

# Wait for cluster to be ready
sleep 10

# Run benchmarks
docker-compose -f docker-compose.ray.yml run --rm benchmark

# View Ray dashboard
# Open http://localhost:8265 in browser

# Stop cluster
docker-compose -f docker-compose.ray.yml down
```

## Configuration

The container expects `config.yaml` to be mounted at `/workspace/config.yaml`.

For Ray distributed execution, ensure:
- `ray.enabled: true` in config.yaml, OR
- Use `--distributed` flag

## Volume Mounts

- **Config**: `-v $(pwd)/config.yaml:/workspace/config.yaml:ro`
- **Results**: `-v $(pwd)/results:/workspace/results`

## GPU Access

Ensure GPU access is enabled:
- `--gpus all` flag for Docker
- `nvidia-container-toolkit` installed
- Verify with: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

## Troubleshooting

### GPU not detected in container

```bash
# Verify NVIDIA runtime
docker info | grep nvidia

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Ray connection issues

- Check Ray head is running: `docker ps | grep ray-head`
- Verify network connectivity between containers
- Check Ray logs: `docker logs ray-head`

### Build failures

- Ensure Docker has enough disk space
- Check internet connectivity for downloading packages
- Verify base image is available: `docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`

## Next Steps

For Kubernetes deployment, see `k8s/README.md` (after Phase 6.2 implementation).

