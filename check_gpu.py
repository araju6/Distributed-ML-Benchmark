#!/usr/bin/env python3
"""
GPU diagnostic script
"""
import torch
import os
import subprocess

print("="*70)
print("GPU DIAGNOSTICS")
print("="*70)
print()

print("1. PyTorch CUDA Support:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA compiled: {torch.version.cuda}")
print(f"   cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
print()

print("2. CUDA Availability:")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"   Compute capability: {props.major}.{props.minor}")
    print(f"   Total memory: {props.total_memory / 1e9:.2f} GB")
else:
    print("   ⚠ GPU not detected!")
print()

print("3. Environment Variables:")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
ld_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
print(f"   LD_LIBRARY_PATH: {ld_path[:100] if isinstance(ld_path, str) and len(ld_path) > 100 else ld_path}")
print()

print("4. Hardware Check:")
try:
    result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
    nvidia_devices = [line for line in result.stdout.split('\n') if 'nvidia' in line.lower() or 'NVIDIA' in line]
    if nvidia_devices:
        print("   GPU devices found:")
        for dev in nvidia_devices:
            print(f"     {dev}")
    else:
        print("   No NVIDIA devices found in lspci")
except Exception as e:
    print(f"   Could not run lspci: {e}")
print()

print("5. CUDA Runtime Libraries:")
cuda_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
if os.path.exists(cuda_lib_path):
    cuda_libs = [f for f in os.listdir(cuda_lib_path) if 'cuda' in f.lower() or 'cudnn' in f.lower()]
    print(f"   Found {len(cuda_libs)} CUDA-related libraries in PyTorch")
    print(f"   Path: {cuda_lib_path}")
else:
    print("   Could not find CUDA libraries in PyTorch installation")
print()

print("6. Recommendations:")
if not torch.cuda.is_available():
    print("   ❌ GPU is not accessible. Possible fixes:")
    print("      1. Install nvidia-driver: sudo apt install nvidia-driver-XXX")
    print("      2. Check if GPU is allocated: Check with your cluster admin")
    print("      3. Load nvidia module (if on cluster): module load cuda")
    print("      4. Verify nvidia-smi works (may need to install nvidia-utils)")
else:
    print("   ✓ GPU is accessible and ready to use!")
print("="*70)



