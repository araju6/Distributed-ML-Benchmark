# Install NVIDIA Driver for GPU Access

## Current Situation
- ✅ GPU Hardware: Tesla P100 detected
- ✅ NVIDIA Utilities: nvidia-utils-535 installed
- ❌ NVIDIA Driver Kernel Module: **NOT INSTALLED**
- ❌ Kernel: 5.15.0-160-generic (needs driver module)

## The Fix

You need to install the NVIDIA driver package that includes the kernel module.

### Option 1: Install Specific Driver Version (Recommended)

Since you have `nvidia-utils-535` installed, install the matching driver:

```bash
sudo apt update
sudo apt install nvidia-driver-535
```

After installation, **reboot** the system:
```bash
sudo reboot
```

After reboot, verify:
```bash
nvidia-smi
python check_gpu.py
```

### Option 2: Auto-Install Recommended Driver

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Option 3: Check Available Drivers

```bash
ubuntu-drivers devices
# Or
apt-cache search nvidia-driver | grep "^nvidia-driver"
```

Then install one:
```bash
sudo apt install nvidia-driver-XXX  # Replace XXX with version
sudo reboot
```

## Why Reboot is Needed

The NVIDIA driver kernel module needs to be loaded at boot time. After installation:
1. The kernel module will be compiled for your kernel (5.15.0-160-generic)
2. The system will configure it to load on boot
3. Reboot ensures the module loads properly

## Verification After Reboot

```bash
# Check if module is loaded
lsmod | grep nvidia

# Should show:
# nvidia_uvm
# nvidia_drm
# nvidia_modeset
# nvidia

# Check GPU
nvidia-smi

# Check PyTorch
python check_gpu.py
```

## If You Don't Have Sudo Access

Contact your system administrator with this information:
- Kernel version: `5.15.0-160-generic`
- nvidia-utils-535 is installed but driver module is missing
- Request: "Please install nvidia-driver-535 kernel module for kernel 5.15.0-160-generic"

## Alternative: DKMS

If the driver package doesn't work, you may need DKMS to compile the module:
```bash
sudo apt install dkms
sudo apt install nvidia-driver-535
```


