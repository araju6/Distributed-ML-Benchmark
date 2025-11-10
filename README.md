# Final Project - Distributed ML Benchmark

ML Compiler Benchmark Framework

## Setup

### Option 1: Local Installation (Conda)

**First Time (One-time installation):**

```bash
cd /path/to/Distributed-ML-Bench
./setup.sh
```

This installs conda and creates the environment (~5-15 minutes).

**Every Session:**

```bash
cd /path/to/Distributed-ML-Bench
source startup.sh
```

Or manually:
```bash
conda activate ml-benchmark
```

### Option 2: Docker Containerization

**Build the Docker image:**

```bash
./scripts/docker-build.sh
```

**Run with Docker:**

```bash
# Sequential execution
docker run --rm --gpus all \
  -v $(pwd)/config.yaml:/workspace/config.yaml:ro \
  -v $(pwd)/results:/workspace/results \
  ml-benchmark:latest

# Distributed execution with Ray
docker run --rm --gpus all \
  -v $(pwd)/config.yaml:/workspace/config.yaml:ro \
  -v $(pwd)/results:/workspace/results \
  ml-benchmark:latest \
  python /workspace/run_benchmark.py --distributed
```

**Or use Docker Compose:**

```bash
# Single container
docker-compose up

# Multi-container Ray cluster
docker-compose -f docker-compose.ray.yml up

# Full stack (Ray + Monitoring)
docker-compose -f docker-compose.full.yml up
```

See `docker/README.md` for detailed Docker instructions.

### Option 3: Kubernetes Deployment (Production)

**Prerequisites:**
- Kubernetes cluster with GPU nodes
- KubeRay operator installed
- NVIDIA GPU device plugin

**Deploy to Kubernetes:**

```bash
# Deploy RayCluster and all resources
./scripts/k8s-deploy.sh

# Submit benchmark job
./scripts/k8s-submit-job.sh

# Access Ray dashboard
kubectl port-forward svc/ml-benchmark-cluster-head-svc 8265:8265 -n ml-benchmark
```

See `k8s/README.md` for detailed Kubernetes deployment instructions.

## Running

### Run Benchmarks

**Sequential execution** (default):
```bash
python run_benchmark.py
```

**Distributed execution** (multi-GPU parallel with Ray):
```bash
python run_benchmark.py --distributed
```

Or enable in `config.yaml`:
```yaml
ray:
  enabled: true
```

Results saved to `results/benchmark_results.csv`.

### Distributed Execution with Ray

Ray enables parallel execution across multiple GPUs on a single node. Each GPU runs a different benchmark task concurrently, providing 4-8× speedup on multi-GPU nodes.

**Features:**
- Automatic GPU detection and assignment
- GPU isolation per task (via `CUDA_VISIBLE_DEVICES`)
- Round-robin task distribution across GPUs
- Support for connecting to existing Ray clusters (e.g., Kubernetes)

**Configuration:**
- Set `ray.enabled: true` in `config.yaml` or use `--distributed` flag
- Configure `ray.num_gpus` to specify GPU count (auto-detect if null)
- Set `ray.head_address` to connect to existing cluster (for K8s)

### Prometheus Metrics Export

Prometheus metrics are automatically exported when enabled (default: enabled):

```yaml
# In config.yaml
monitoring:
  prometheus:
    enabled: true
    port: 8000  # HTTP port for metrics endpoint
```

Access metrics at: `http://localhost:8000/metrics`

**Available Metrics:**
- `benchmark_latency_seconds` - Inference latency histogram (per iteration)
- `benchmark_throughput_samples_per_sec` - Throughput gauge
- `benchmark_gpu_memory_mb` - GPU memory usage (peak and avg)
- `benchmark_compile_time_seconds` - Compilation time histogram
- `benchmark_runs_total` - Counter of completed benchmark runs
- `benchmark_iterations_total` - Counter of total iterations executed

All metrics include labels: `compiler`, `model`, `batch_size`, `gpu_id`

**Grafana Dashboard:**
A pre-configured Grafana dashboard is available for visualization:
```bash
# Start monitoring stack (Prometheus + Grafana)
cd monitoring
docker-compose up -d

# Access Grafana: http://localhost:3000 (admin/admin)
# Dashboard: "ML Compiler Benchmark Dashboard"
```

See `monitoring/README.md` for detailed setup instructions.

### NVIDIA Nsight Systems Profiling

Enable kernel-level CUDA profiling with Nsight Systems:

```yaml
# In config.yaml
profiling:
  enabled: true
  output_dir: results/profiles
  profile_iterations: 10  # Number of iterations to profile
```

Profiles are saved as `.nsys-rep` files. View with:
```bash
nsys-ui results/profiles/<model>_<compiler>_bs<batch_size>.nsys-rep
```

### AutoCompiler Test Mode

AutoCompiler runs benchmarks across all available compilers in parallel and generates a comparison report to help you choose the best compiler for your model.

**Usage:**
```bash
# Test ResNet-50 with batch size 32
python run_autocompiler.py --model resnet50 --input-shape 3 224 224 --batch-size 32

# Test BERT with sequence length 128
python run_autocompiler.py --model bert_base --max-length 128 --batch-size 8

# Test specific compilers only
python run_autocompiler.py --model resnet50 --input-shape 3 224 224 --batch-size 32 \
  --compilers pytorch_eager torchscript onnx_runtime

# Sequential execution (no Ray)
python run_autocompiler.py --model resnet50 --input-shape 3 224 224 --batch-size 32 --no-ray
```

**Features:**
- Parallel execution across all compilers (uses Ray if available)
- Comprehensive comparison report with recommendations
- Best compiler recommendations for:
  - Lowest latency
  - Highest throughput
  - Best memory efficiency
  - Balanced performance
- JSON report export for further analysis

**Output:**
- Console report with recommendations
- JSON report saved to `results/autocompiler_report.json` (configurable)

### Analyze Results

```bash
python analyze_results.py
```

## Configuration

Edit `config.yaml` to change:
- Models: Multiple models can be benchmarked (vision and NLP)
- Compilers: Choose from available compilers
- Batch sizes: Configure per model

### Available Compilers

**PyTorch Compilers:**
- `pytorch_eager` - Baseline (no compilation, reference implementation)
- `torchscript` - TorchScript JIT compiler (works on P100!)
- `torch_inductor` - TorchInductor (requires newer GPU, doesn't work on P100)

**Cross-Platform Compilers:**
- `onnx_runtime` - ONNX Runtime with GPU support (CUDAExecutionProvider)

**TVM Compilers** (requires `apache-tvm` installation):
- `tvm` - Apache TVM compiler (standard optimization)
- `tvm_autotuned` - TVM with autotuning (slower compilation, better performance)

**TensorRT Compilers** (requires NVIDIA TensorRT installation):
- `tensorrt` or `tensorrt_fp32` - TensorRT FP32 (standard precision)
- `tensorrt_fp16` - TensorRT FP16 (faster, may have accuracy loss)
- Note: TensorRT may have limitations on P100 (compute capability 6.0)

### Available Models

**Vision Models:**
- `resnet50` - ResNet-50 (ImageNet)
- `mobilenet_v3_large` - MobileNetV3 Large (ImageNet)

**NLP Models:**
- `bert_base` - BERT-base-uncased (HuggingFace)
- `gpt2` - GPT-2 (HuggingFace)

See `config.yaml` for configuration examples.
