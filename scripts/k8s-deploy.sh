#!/bin/bash
# Kubernetes deployment script for ML Benchmark
# Deploys namespace, ConfigMap, PVC, and RayCluster

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$PROJECT_DIR/k8s"

cd "$PROJECT_DIR"

echo "======================================================================"
echo "Deploying ML Benchmark to Kubernetes"
echo "======================================================================"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if KubeRay operator is installed
if ! kubectl get crd rayclusters.ray.io &> /dev/null; then
    echo "Error: KubeRay operator is not installed"
    echo ""
    echo "Install KubeRay operator with:"
    echo "  kubectl create -k https://github.com/ray-project/kuberay/ray-operator/config/default"
    echo ""
    echo "Or:"
    echo "  kubectl apply -f https://github.com/ray-project/kuberay/releases/latest/download/kuberay-operator.yaml"
    exit 1
fi

echo "✓ KubeRay operator detected"
echo ""

# Step 1: Create namespace
echo "Step 1: Creating namespace..."
kubectl apply -f "$K8S_DIR/namespace.yaml"
kubectl wait --for=condition=Active namespace/ml-benchmark --timeout=30s
echo "✓ Namespace created"
echo ""

# Step 2: Create ConfigMap
echo "Step 2: Creating ConfigMap..."
kubectl apply -f "$K8S_DIR/configmap.yaml"
echo "✓ ConfigMap created"
echo ""

# Step 3: Create PVC
echo "Step 3: Creating PersistentVolumeClaim..."
kubectl apply -f "$K8S_DIR/pvc.yaml"
echo "✓ PVC created"
echo ""

# Step 4: Deploy RayCluster
echo "Step 4: Deploying RayCluster..."
echo "Note: Update image name in k8s/raycluster.yaml if using a registry"
kubectl apply -f "$K8S_DIR/raycluster.yaml"
echo "✓ RayCluster deployed"
echo ""

# Wait for RayCluster to be ready
echo "Waiting for RayCluster to be ready..."
echo "This may take a few minutes while pods are being created..."
kubectl wait --for=condition=ready pod -l ray.io/cluster=ml-benchmark-cluster -n ml-benchmark --timeout=300s || true

# Check cluster status
echo ""
echo "RayCluster status:"
kubectl get raycluster ml-benchmark-cluster -n ml-benchmark

echo ""
echo "Pod status:"
kubectl get pods -n ml-benchmark -l ray.io/cluster=ml-benchmark-cluster

echo ""
echo "======================================================================"
echo "✓ Deployment complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Check Ray dashboard:"
echo "     kubectl port-forward svc/ml-benchmark-cluster-head-svc 8265:8265 -n ml-benchmark"
echo "     Then open http://localhost:8265 in your browser"
echo ""
echo "  2. Submit a benchmark job:"
echo "     kubectl apply -f k8s/rayjob.yaml"
echo ""
echo "  3. Check job status:"
echo "     kubectl get rayjob -n ml-benchmark"
echo ""
echo "  4. View job logs:"
echo "     kubectl logs -f rayjob/benchmark-job -n ml-benchmark"
echo ""

