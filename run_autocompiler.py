#!/usr/bin/env python
"""
AutoCompiler Test Mode Runner

Runs benchmarks across all available compilers in parallel and generates
a comparison report to help users choose the best compiler for their model.
"""

import argparse
import torch
from benchmark.core.config import Config
from benchmark.core.autocompiler import AutoCompiler
from benchmark.models.resnet import ResNetWrapper
from benchmark.models.mobilenet import MobileNetWrapper
from benchmark.models.bert import BERTWrapper
from benchmark.models.gpt2 import GPT2Wrapper


def get_model(model_name: str, input_format: dict):
    """Get model wrapper from name and input format."""
    if model_name == "resnet50":
        if "input_shape" not in input_format:
            raise ValueError("resnet50 requires input_shape")
        return ResNetWrapper(input_shape=tuple(input_format["input_shape"]), pretrained=True)
    elif model_name == "mobilenet_v3_large":
        if "input_shape" not in input_format:
            raise ValueError("mobilenet_v3_large requires input_shape")
        return MobileNetWrapper(input_shape=tuple(input_format["input_shape"]), pretrained=True)
    elif model_name == "bert_base":
        if "max_length" not in input_format:
            raise ValueError("bert_base requires max_length")
        return BERTWrapper(max_length=input_format["max_length"], pretrained=True)
    elif model_name == "gpt2":
        if "max_length" not in input_format:
            raise ValueError("gpt2 requires max_length")
        return GPT2Wrapper(max_length=input_format["max_length"], pretrained=True)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: resnet50, mobilenet_v3_large, bert_base, gpt2"
        )


def main():
    parser = argparse.ArgumentParser(
        description='AutoCompiler Test Mode - Compare all compilers for your model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test ResNet-50 with batch size 32
  python run_autocompiler.py --model resnet50 --input-shape 3 224 224 --batch-size 32
  
  # Test BERT with sequence length 128
  python run_autocompiler.py --model bert_base --max-length 128 --batch-size 8
  
  # Test with specific compilers only
  python run_autocompiler.py --model resnet50 --input-shape 3 224 224 --batch-size 32 \\
    --compilers pytorch_eager torchscript onnx_runtime
  
  # Sequential execution (no Ray)
  python run_autocompiler.py --model resnet50 --input-shape 3 224 224 --batch-size 32 --no-ray
        """
    )
    
    parser.add_argument(
        '--model',
        required=True,
        choices=['resnet50', 'mobilenet_v3_large', 'bert_base', 'gpt2'],
        help='Model to test'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        required=True,
        help='Batch size for testing'
    )
    
    # Input format options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-shape',
        nargs='+',
        type=int,
        help='Input shape for vision models (e.g., 3 224 224 for ImageNet)'
    )
    input_group.add_argument(
        '--max-length',
        type=int,
        help='Max sequence length for NLP models (e.g., 128)'
    )
    
    parser.add_argument(
        '--compilers',
        nargs='+',
        help='Specific compilers to test (default: all available)'
    )
    
    parser.add_argument(
        '--warmup-iters',
        type=int,
        default=10,
        help='Number of warmup iterations (default: 10)'
    )
    
    parser.add_argument(
        '--measured-iters',
        type=int,
        default=50,
        help='Number of measured iterations (default: 50, reduced for quick testing)'
    )
    
    parser.add_argument(
        '--no-ray',
        action='store_true',
        help='Disable Ray parallel execution (use sequential)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/autocompiler_report.json',
        help='Output path for JSON report (default: results/autocompiler_report.json)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (for warmup/measured iterations, default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Parse input format
    input_format = {}
    if args.input_shape:
        input_format['input_shape'] = args.input_shape
    if args.max_length:
        input_format['max_length'] = args.max_length
    
    # Try to load config for default values
    try:
        cfg = Config.from_yaml(args.config)
        warmup_iters = args.warmup_iters or cfg.benchmark.warmup_iterations
        measured_iters = args.measured_iters or cfg.benchmark.measured_iterations
    except Exception:
        warmup_iters = args.warmup_iters
        measured_iters = args.measured_iters
    
    # Get model wrapper
    model_wrapper = get_model(args.model, input_format)
    
    # Initialize AutoCompiler
    autocompiler = AutoCompiler(
        warmup_iters=warmup_iters,
        measured_iters=measured_iters,
        use_ray=not args.no_ray
    )
    
    # Test all compilers
    results = autocompiler.test_all_compilers(
        model_wrapper=model_wrapper,
        batch_size=args.batch_size,
        compiler_filter=args.compilers
    )
    
    # Generate report
    report = autocompiler.generate_report(
        model_wrapper=model_wrapper,
        batch_size=args.batch_size,
        results=results
    )
    
    # Print report
    autocompiler.print_report(report)
    
    # Save report
    autocompiler.save_report(report, args.output)
    
    print(f"\n✓ AutoCompiler test complete!")
    print(f"  Report saved to: {args.output}")
    print(f"  Use the recommendations above to choose the best compiler for your use case.\n")


if __name__ == "__main__":
    main()

