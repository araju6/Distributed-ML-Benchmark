# Multi-stage Dockerfile for ML Compiler Benchmark
# Base: CUDA runtime for P100 GPUs (compute capability 6.0)

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:${PATH}

# Set working directory
WORKDIR /workspace

# Copy environment file
COPY environment.yml /workspace/

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Make conda environment available
ENV PATH=/opt/conda/envs/ml-benchmark/bin:${PATH}
ENV CONDA_DEFAULT_ENV=ml-benchmark

# Copy benchmark code
COPY benchmark/ /workspace/benchmark/
COPY run_benchmark.py /workspace/
COPY analyze_results.py /workspace/
COPY config.yaml /workspace/

# Copy scripts
COPY scripts/ /workspace/scripts/
RUN chmod +x /workspace/scripts/*.sh

# Create results directory
RUN mkdir -p /workspace/results

# Set entrypoint
ENTRYPOINT ["/workspace/scripts/ray-entrypoint.sh"]

# Default command (can be overridden)
CMD ["python", "/workspace/run_benchmark.py"]

