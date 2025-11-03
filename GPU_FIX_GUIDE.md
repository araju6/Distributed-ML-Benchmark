# GPU Fix Guide

## Current Situation
- ✅ **GPU Hardware:** Tesla P100 is present and detected
- ✅ **PyTorch:** Compiled with CUDA 11.8 support
- ❌ **Driver:** NVIDIA driver kernel module is NOT loaded
- ❌ **Access:** `/dev/nvidia*` devices don't exist

## The Problem
The NVIDIA driver needs to be loaded by the system/kernel. This typically requires:
- Either: System admin to install/load drivers
- Or: Proper cluster/cloud GPU allocation

## Solutions (Try in Order)

### Solution 1: Request GPU Allocation (Cluster/Cloud)
If you're on a cluster or cloud VM, you may need to request GPU access:

```bash
# Check if you need to request GPU via scheduler
# For SLURM clusters:
# sbatch --gpus=1 script.sh
# or
# srun --gpus=1 bash

# For cloud providers, check your VM configuration
```

**Action:** Contact your cluster/cloud admin and ask:
- "How do I enable GPU access on this node?"
- "Is the GPU allocated to my session?"
- "Do I need to use a specific queue/partition?"

---

### Solution 2: Install NVIDIA Drivers (Requires Admin/Sudo)
If you have sudo access or can request admin help:

```bash
# 1. Check available driver versions
ubuntu-drivers devices

# 2. Install recommended driver
sudo ubuntu-drivers autoinstall

# OR install specific version
sudo apt update
sudo apt install nvidia-driver-535  # or appropriate version

# 3. Install nvidia-utils for nvidia-smi
sudo apt install nvidia-utils-535

# 4. Reboot (if required)
sudo reboot

# 5. Verify after reboot
nvidia-smi
```

**Action:** 
- If you have sudo: Run the commands above
- If not: Share these commands with your system admin

---

### Solution 3: Check if Driver Needs Loading
Sometimes drivers are installed but not loaded:

```bash
# Check if nvidia module exists but isn't loaded
lsmod | grep nvidia

# Try to manually load (may need sudo)
sudo modprobe nvidia

# Check if it loaded
lsmod | grep nvidia
```

---

### Solution 4: Environment Variables (If Driver is Installed Elsewhere)
If CUDA/drivers are installed in a non-standard location:

```bash
# Find CUDA installation
find /usr/local -name "libcuda.so*" 2>/dev/null
find /opt -name "libcuda.so*" 2>/dev/null

# If found, add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH
```

---

## Verification Steps

After attempting any solution, verify:

```bash
# 1. Check if driver is loaded
cat /proc/driver/nvidia/version

# 2. Check if nvidia-smi works
nvidia-smi

# 3. Check PyTorch GPU access
python check_gpu.py

# 4. Run a quick test
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Expected Results After Fix

Once GPU is accessible, you should see:
- `nvidia-smi` shows the P100 GPU
- `python check_gpu.py` shows `CUDA available: True`
- Benchmark automatically uses GPU (check output for "Using GPU: Tesla P100")
- Much faster performance (5-10x speedup vs CPU)

---

## Quick Diagnostic

Run this to see current status:
```bash
./fix_gpu.sh
python check_gpu.py
```

---

## Contact Info

If none of these work, contact your system administrator with:
1. Output of `lspci | grep -i nvidia`
2. Output of `./fix_gpu.sh`
3. Request: "I need GPU access on this node. The P100 is detected but drivers aren't loaded."



