#!/bin/bash
# Ray entrypoint script for Docker containers
# Handles Ray head/worker modes or direct benchmark execution

set -e

# Check if we're running as Ray head or worker
if [ "$RAY_ROLE" = "head" ]; then
    # Running as Ray head
    echo "Starting Ray head node"
    ray start --head --block
elif [ "$RAY_ROLE" = "worker" ] || [ -n "$RAY_HEAD_ADDRESS" ]; then
    # Running as Ray worker - connect to head
    if [ -z "$RAY_HEAD_ADDRESS" ]; then
        echo "ERROR: RAY_ROLE=worker but RAY_HEAD_ADDRESS not set"
        exit 1
    fi
    echo "Starting Ray worker, connecting to head at $RAY_HEAD_ADDRESS"
    ray start --address="$RAY_HEAD_ADDRESS" --block
else
    # Not running Ray - execute command directly (benchmark or other)
    if [ $# -eq 0 ]; then
        # No arguments provided, use default command
        exec python /workspace/run_benchmark.py
    else
        # Execute provided command
        exec "$@"
    fi
fi

