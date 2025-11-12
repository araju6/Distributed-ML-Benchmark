"""Standalone script for Nsight Systems profiling of a single benchmark.

This script is designed to be run with nsys:
    nsys profile --trace=cuda,nvtx,osrt --output=profile.nsys-rep python -m benchmark.core.profile_benchmark <args>

Or use the integrated profiling in benchmark_runner.py
"""

import sys
import os
import torch
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.utils.device import get_device
from benchmark.utils.factories import get_compiler, get_model

def main():
    parser = argparse.ArgumentParser(description='Profile a single benchmark with Nsight Systems')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--compiler', required=True, help='Compiler name')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations to profile')
    parser.add_argument('--input-shape', nargs='+', type=int, help='Input shape for vision models')
    parser.add_argument('--max-length', type=int, help='Max length for NLP models')
    
    args = parser.parse_args()
    
    device = get_device()
    
    # Prepare model config
    model_config = {}
    if args.input_shape:
        model_config['input_shape'] = args.input_shape
    if args.max_length:
        model_config['max_length'] = args.max_length
    
    # Get model and compiler
    model_wrapper = get_model(args.model, input_format=model_config)
    compiler = get_compiler(args.compiler)
    
    # Compile model
    model = model_wrapper.get_model().to(device)
    example_input = model_wrapper.get_example_input(args.batch_size, device)
    compiled_model = compiler.compile(model, example_input)
    
    # Run profiling iterations
    print(f"Profiling {args.model} | {args.compiler} | batch_size={args.batch_size}")
    print(f"Running {args.iterations} iterations...")
    
    with torch.no_grad():
        for i in range(args.iterations):
            _ = compiled_model(example_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{args.iterations} iterations")
    
    print("Profiling complete!")

if __name__ == "__main__":
    main()

