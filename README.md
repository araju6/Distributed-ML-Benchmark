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
- Model: `resnet50`
- Compilers: `pytorch_eager`, `torchscript`
- Batch sizes: `[1, 32]`
