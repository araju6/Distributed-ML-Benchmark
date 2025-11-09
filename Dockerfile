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

# Install NVIDIA Nsight Systems for profiling
# Note: nsys is typically installed with CUDA toolkit, but we install it explicitly
# for containerized environments where it might not be available
# Try to install latest available version, or skip if not available (for non-GPU builds)
RUN apt-get update && \
    (apt-get install -y --no-install-recommends nsight-systems-2025.3.2 2>/dev/null || \
     apt-get install -y --no-install-recommends nsight-systems-2025.1.3 2>/dev/null || \
     apt-get install -y --no-install-recommends nsight-systems-2024.6.2 2>/dev/null || \
     echo "Warning: Nsight Systems not available, profiling will be disabled") && \
    rm -rf /var/lib/apt/lists/*

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

# Accept Conda Terms of Service and create conda environment
# Configure conda for better network reliability
RUN conda config --set remote_connect_timeout_secs 60.0 && \
    conda config --set remote_read_timeout_secs 120.0 && \
    conda config --set remote_max_retries 5 && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment with retry logic for network issues
RUN set +e && \
    SUCCESS=0 && \
    for i in 1 2 3; do \
        echo "Attempt $i to create conda environment..." && \
        conda env create -f environment.yml && \
        if [ $? -eq 0 ]; then SUCCESS=1; break; fi && \
        echo "Attempt $i failed, waiting 10 seconds before retry..." && \
        sleep 10; \
    done && \
    set -e && \
    if [ $SUCCESS -eq 0 ]; then \
        echo "ERROR: Failed to create conda environment after 3 attempts" && \
        exit 1; \
    fi && \
    conda clean -afy

# Make conda environment available
ENV PATH=/opt/conda/envs/ml-benchmark/bin:${PATH}
ENV CONDA_DEFAULT_ENV=ml-benchmark

# Copy benchmark code
COPY benchmark/ /workspace/benchmark/
COPY run_benchmark.py /workspace/
COPY run_autocompiler.py /workspace/
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

