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

```bash
python run_benchmark.py
```

Results saved to `results/benchmark_results.csv`.

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
