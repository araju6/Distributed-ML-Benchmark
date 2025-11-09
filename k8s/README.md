# Kubernetes Deployment Guide

This guide explains how to deploy the ML Compiler Benchmark Framework to Kubernetes using KubeRay operator for distributed execution.

## Prerequisites

### 1. Kubernetes Cluster
- Kubernetes cluster (v1.20+) with GPU nodes
- Access to cluster via `kubectl`
- Cluster admin permissions (for namespace/CRD creation)

### 2. KubeRay Operator
Install the KubeRay operator to manage Ray clusters:

```bash
# Option 1: Install from KubeRay repository
kubectl create -k https://github.com/ray-project/kuberay/ray-operator/config/default

# Option 2: Install from release
kubectl apply -f https://github.com/ray-project/kuberay/releases/latest/download/kuberay-operator.yaml

# Verify installation
kubectl get crd rayclusters.ray.io
kubectl get pods -n ray-system
```

### 3. GPU Device Plugin
Install NVIDIA GPU device plugin for GPU access:

```bash
# Option 1: NVIDIA GPU Operator (recommended)
# Follow: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html

# Option 2: NVIDIA Device Plugin DaemonSet
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpu: .status.capacity."nvidia.com/gpu"}'
```

### 4. Container Registry
- Docker image built and pushed to accessible registry
- Or use local image (requires `imagePullPolicy: Never`)

## Quick Start

### Step 1: Build and Push Docker Image

```bash
# Build image
./scripts/docker-build.sh

# Tag for Docker Hub
docker tag ml-benchmark:latest 5ifty6ix56/ml-benchmark:latest

# Push to Docker Hub
docker push 5ifty6ix56/ml-benchmark:latest

# Or use the build script with push:
DOCKERHUB_USERNAME=5ifty6ix56 ./scripts/docker-build.sh latest --push

# Update image in k8s/raycluster.yaml and k8s/rayjob.yaml
```

### Step 2: Deploy to Kubernetes

```bash
# Automated deployment
./scripts/k8s-deploy.sh

# Or manual deployment
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/raycluster.yaml
kubectl apply -f k8s/service-metrics.yaml
# Optional: If using Prometheus Operator
kubectl apply -f k8s/servicemonitor.yaml
```

### Step 3: Verify Deployment

```bash
# Check RayCluster status
kubectl get raycluster -n ml-benchmark

# Check pods
kubectl get pods -n ml-benchmark -l ray.io/cluster=ml-benchmark-cluster

# Check services
kubectl get svc -n ml-benchmark
```

### Step 4: Access Ray Dashboard

```bash
# Port forward Ray dashboard
kubectl port-forward svc/ml-benchmark-cluster-head-svc 8265:8265 -n ml-benchmark

# Open http://localhost:8265 in browser
```

### Step 5: Submit Benchmark Job

```bash
# Automated submission
./scripts/k8s-submit-job.sh

# Or manual submission
kubectl apply -f k8s/rayjob.yaml
```

## Configuration

### RayCluster Configuration

Edit `k8s/raycluster.yaml` to customize:

- **Worker replicas**: Change `replicas` in `workerGroupSpecs` (default: 4)
- **GPU resources**: Ensure `nvidia.com/gpu: "1"` is set per worker
- **CPU/Memory**: Adjust `resources.requests` and `resources.limits`
- **Image**: Update `image` field with your registry path

### Benchmark Configuration

Edit `k8s/configmap.yaml` to change:
- Models to benchmark
- Compilers to test
- Batch sizes
- Iteration counts

Or update ConfigMap directly:
```bash
kubectl edit configmap benchmark-config -n ml-benchmark
```

### Scaling Workers

**Manual scaling:**
```bash
# Edit raycluster.yaml and change replicas
kubectl apply -f k8s/raycluster.yaml
```

**Auto-scaling** (if enabled):
```bash
# Edit raycluster.yaml and set enableInTreeAutoscaling: true
# Configure minReplicas and maxReplicas
kubectl apply -f k8s/raycluster.yaml
```

## Monitoring

### Prometheus Metrics

The benchmark runner exposes Prometheus metrics on port 8000. Metrics are automatically collected when enabled in the ConfigMap.

**Deploy Metrics Service and ServiceMonitor:**

```bash
# Deploy metrics service
kubectl apply -f k8s/service-metrics.yaml

# Deploy ServiceMonitor (requires Prometheus Operator)
kubectl apply -f k8s/servicemonitor.yaml
```

**Verify metrics endpoint:**

```bash
# Port forward to a running pod
kubectl port-forward <pod-name> 8000:8000 -n ml-benchmark

# Check metrics
curl http://localhost:8000/metrics
```

**Prometheus Configuration:**

If using Prometheus Operator, the ServiceMonitor will automatically configure scraping. Otherwise, add to your Prometheus config:

```yaml
scrape_configs:
  - job_name: 'ml-benchmark'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ml-benchmark
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: ml-compiler-benchmark
      - source_labels: [__meta_kubernetes_pod_label_job_type]
        action: keep
        regex: benchmark
      - source_labels: [__meta_kubernetes_pod_ip]
        target_label: __address__
        replacement: ${1}:8000
```

### Ray Dashboard
- Access via port-forward: `kubectl port-forward svc/ml-benchmark-cluster-head-svc 8265:8265 -n ml-benchmark`
- View cluster status, jobs, and resource utilization

### Job Status
```bash
# List jobs
kubectl get rayjob -n ml-benchmark

# Job details
kubectl describe rayjob benchmark-job -n ml-benchmark

# Job logs
kubectl logs -f rayjob/benchmark-job -n ml-benchmark
```

### Pod Logs
```bash
# All pods in cluster
kubectl logs -f -l ray.io/cluster=ml-benchmark-cluster -n ml-benchmark

# Specific pod
kubectl logs -f <pod-name> -n ml-benchmark
```

## Accessing Results

### From Pod
```bash
# List result files
kubectl exec -it <pod-name> -n ml-benchmark -- ls -la /workspace/results

# View CSV
kubectl exec -it <pod-name> -n ml-benchmark -- cat /workspace/results/benchmark_results.csv

# Copy results locally
kubectl cp ml-benchmark/<pod-name>:/workspace/results/benchmark_results.csv ./results/
```

### From PVC
If using persistent volume:
```bash
# Create a pod to access PVC
kubectl run -it --rm debug --image=busybox --restart=Never -n ml-benchmark -- sh
# Inside pod: ls /mnt/results
```

## Troubleshooting

### GPU Not Available

```bash
# Check GPU nodes
kubectl get nodes -o json | jq '.items[] | select(.status.capacity."nvidia.com/gpu")'

# Check device plugin
kubectl get pods -n kube-system | grep nvidia

# Verify GPU in pod
kubectl exec -it <pod-name> -n ml-benchmark -- nvidia-smi
```

### Ray Connection Failed

```bash
# Check head service
kubectl get svc -n ml-benchmark

# Check head pod logs
kubectl logs -f <head-pod-name> -n ml-benchmark

# Verify DNS resolution
kubectl exec -it <worker-pod-name> -n ml-benchmark -- nslookup ml-benchmark-cluster-head-svc.ml-benchmark.svc.cluster.local
```

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n ml-benchmark

# Check events
kubectl get events -n ml-benchmark --sort-by='.lastTimestamp'

# Check resource quotas
kubectl describe quota -n ml-benchmark
```

### Out of Memory

```bash
# Check resource usage
kubectl top pods -n ml-benchmark

# Increase memory limits in raycluster.yaml
# Update resources.requests.memory and resources.limits.memory
```

### Image Pull Errors

```bash
# For local images, use imagePullPolicy: Never
# Update in raycluster.yaml:
# imagePullPolicy: Never

# For registry images, check:
# - Image exists in registry
# - Registry credentials configured (imagePullSecrets)
# - Network connectivity
```

## Cleanup

```bash
# Delete RayJob
kubectl delete rayjob benchmark-job -n ml-benchmark

# Delete RayCluster
kubectl delete raycluster ml-benchmark-cluster -n ml-benchmark

# Delete all resources
kubectl delete namespace ml-benchmark

# Or delete individually
kubectl delete -f k8s/rayjob.yaml
kubectl delete -f k8s/servicemonitor.yaml  # If deployed
kubectl delete -f k8s/service-metrics.yaml
kubectl delete -f k8s/raycluster.yaml
kubectl delete -f k8s/pvc.yaml
kubectl delete -f k8s/configmap.yaml
kubectl delete -f k8s/namespace.yaml
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │           RayCluster (KubeRay CRD)               │  │
│  │                                                   │  │
│  │  ┌──────────────┐        ┌──────────────┐        │  │
│  │  │ Ray Head Pod │        │ Worker Pods  │        │  │
│  │  │  (No GPU)    │◄───────┤  (1 GPU each)│        │  │
│  │  │              │        │              │        │  │
│  │  └──────────────┘        └──────────────┘        │  │
│  │         │                          │             │  │
│  └─────────┼──────────────────────────┼─────────────┘  │
│            │                          │                │
│  ┌─────────▼──────────────────────────▼─────────────┐ │
│  │           RayJob (KubeRay CRD)                    │ │
│  │  Submits benchmark tasks to RayCluster            │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │  ConfigMap: benchmark-config                     │ │
│  │  PVC: benchmark-results-pvc                      │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Multi-Node Setup

For multi-node deployment, ensure:
1. All nodes have GPU device plugin installed
2. RayCluster workers can access head service via DNS
3. Network policies allow inter-pod communication
4. Storage class supports ReadWriteMany (for shared PVC)

Update `raycluster.yaml`:
- Increase worker replicas
- Verify node selectors if needed
- Check resource quotas per namespace

## Profiling Support

NVIDIA Nsight Systems profiling is now supported in the container image. To enable profiling:

1. **Update ConfigMap** to enable profiling:
   ```bash
   kubectl edit configmap benchmark-config -n ml-benchmark
   ```
   
   Set:
   ```yaml
   profiling:
     enabled: true
     output_dir: /workspace/results/profiles
     profile_iterations: 10
   ```

2. **Profiles are saved** to the PVC at `/workspace/results/profiles/`

3. **Access profiles**:
   ```bash
   # Copy profile from pod
   kubectl cp ml-benchmark/<pod-name>:/workspace/results/profiles/<profile-name>.nsys-rep ./results/
   
   # View with nsys-ui
   nsys-ui results/<profile-name>.nsys-rep
   ```

**Note**: Profiling adds overhead and increases benchmark time. Use for detailed performance analysis, not routine benchmarking.

## Best Practices

1. **Resource Management**: Set appropriate CPU/memory requests and limits
2. **GPU Isolation**: One GPU per worker pod prevents conflicts
3. **ConfigMap Updates**: Update ConfigMap without rebuilding images
4. **PVC Storage**: Use ReadWriteMany for shared results access
5. **Monitoring**: Set up Prometheus/Grafana for production monitoring (see Monitoring section above)
6. **Profiling**: Enable profiling only when needed for detailed analysis
7. **Logging**: Use centralized logging (e.g., ELK stack) for distributed debugging

## Next Steps

- ✅ Prometheus metrics scraping configured (see Monitoring section)
- Set up Grafana dashboards for monitoring
- Implement auto-scaling based on workload
- Add health checks and liveness probes
- Configure resource quotas and limits
- ✅ Profiling support enabled (see Profiling Support section)

