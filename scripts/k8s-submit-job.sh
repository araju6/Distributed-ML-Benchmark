#!/bin/bash
# Submit benchmark job to Kubernetes RayCluster

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$PROJECT_DIR/k8s"

cd "$PROJECT_DIR"

JOB_NAME="${1:-benchmark-job}"

echo "======================================================================"
echo "Submitting benchmark job to Kubernetes RayCluster"
echo "======================================================================"
echo ""

# Check if RayCluster exists
if ! kubectl get raycluster ml-benchmark-cluster -n ml-benchmark &> /dev/null; then
    echo "Error: RayCluster 'ml-benchmark-cluster' not found in namespace 'ml-benchmark'"
    echo ""
    echo "Deploy the cluster first:"
    echo "  ./scripts/k8s-deploy.sh"
    exit 1
fi

# Check if ConfigMap exists
if ! kubectl get configmap benchmark-config -n ml-benchmark &> /dev/null; then
    echo "Warning: ConfigMap 'benchmark-config' not found. Updating..."
    kubectl apply -f "$K8S_DIR/configmap.yaml"
fi

# Update ConfigMap if config.yaml changed (optional)
read -p "Update ConfigMap from local config.yaml? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Updating ConfigMap..."
    kubectl create configmap benchmark-config \
        --from-file=config.yaml="$PROJECT_DIR/config.yaml" \
        --namespace=ml-benchmark \
        --dry-run=client -o yaml | kubectl apply -f -
    echo "✓ ConfigMap updated"
    echo ""
fi

# Submit RayJob
echo "Submitting RayJob: $JOB_NAME"
kubectl apply -f "$K8S_DIR/rayjob.yaml"

echo ""
echo "Waiting for job to start..."
sleep 5

# Show job status
echo ""
echo "Job status:"
kubectl get rayjob "$JOB_NAME" -n ml-benchmark

echo ""
echo "Job pods:"
kubectl get pods -n ml-benchmark -l ray.io/job-name="$JOB_NAME"

echo ""
echo "======================================================================"
echo "Job submitted!"
echo "======================================================================"
echo ""
echo "Monitor job:"
echo "  kubectl get rayjob $JOB_NAME -n ml-benchmark -w"
echo ""
echo "View logs:"
echo "  kubectl logs -f rayjob/$JOB_NAME -n ml-benchmark"
echo ""
echo "Check results:"
echo "  kubectl exec -it <pod-name> -n ml-benchmark -- cat /workspace/results/benchmark_results.csv"
echo ""

