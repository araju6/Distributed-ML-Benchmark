# Monitoring Setup Guide

This directory contains configuration files for monitoring the ML Benchmark Framework with Prometheus and Grafana.

## Prometheus Setup

### Option 1: Docker Compose (Recommended for Local Testing)

```bash
# Start Prometheus and Grafana
docker-compose up -d

# Access:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (default login: admin/admin)
```

### Option 2: Standalone Prometheus

1. Download Prometheus from https://prometheus.io/download/
2. Use `prometheus.yml` as your configuration
3. Start Prometheus:
   ```bash
   ./prometheus --config.file=monitoring/prometheus.yml
   ```

### Option 3: Kubernetes

Deploy Prometheus using Prometheus Operator or Helm chart, and configure it to scrape the benchmark pods.

## Configuration

### Prometheus Scrape Configuration

The `prometheus.yml` file configures Prometheus to scrape metrics from:
- `localhost:8000` (default benchmark metrics endpoint)

For distributed execution, add additional targets or use service discovery.

### Metrics Endpoint

The benchmark runner exposes metrics at:
- `http://localhost:8000/metrics` (default)

Change the port in `config.yaml`:
```yaml
monitoring:
  prometheus:
    port: 8000
```

## Grafana Dashboards

Pre-configured dashboard is available at `grafana/dashboards/benchmark_dashboard.json`.

The dashboard includes:
- **Inference Latency (p95)** - Latency percentiles by compiler/model
- **Throughput** - Samples per second
- **GPU Memory Usage** - Peak and average memory
- **Compilation Time** - Compiler optimization time
- **Benchmark Statistics** - Run counts and iteration rates

The dashboard is automatically provisioned when using Docker Compose.

## Kubernetes Integration

In Kubernetes, configure Prometheus to scrape metrics from benchmark pods:

```yaml
# Prometheus ServiceMonitor (if using Prometheus Operator)
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-benchmark
  namespace: ml-benchmark
spec:
  selector:
    matchLabels:
      app: ml-compiler-benchmark
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
```

## Quick Start

1. **Start monitoring stack:**
   ```bash
   cd monitoring
   docker-compose up -d
   ```

2. **Run benchmark with metrics enabled:**
   ```bash
   # Ensure monitoring.prometheus.enabled: true in config.yaml
   python run_benchmark.py
   ```

3. **View dashboards:**
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)
   - Metrics endpoint: http://localhost:8000/metrics

The Grafana dashboard will be automatically available after starting the stack.

