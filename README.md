# CS498MLSysProject

ML Compiler Benchmark Framework

## Setup

### First Time (One-time installation)

```bash
cd ~/CS498MLSysProject
./setup.sh
```

This installs conda and creates the environment (~5-15 minutes).

### Every Session

```bash
cd ~/CS498MLSysProject
source startup.sh
```

Or manually:
```bash
conda activate ml-benchmark
```

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
