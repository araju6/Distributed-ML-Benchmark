import os
import torch
from typing import List, Optional

def detect_gpus() -> int:
    """Detect number of available GPUs on current node.
    
    Returns:
        Number of GPUs available (0 if no GPUs)
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def detect_cpus() -> int:
    """Detect number of available CPUs on current node.
    
    Returns:
        Number of CPUs available
    """
    return os.cpu_count() or 1

def get_gpu_assignments(num_tasks: int, num_gpus: Optional[int] = None) -> List[int]:
    """Assign GPU IDs to tasks using round-robin, ensuring each task gets a unique GPU.
    
    For single-node multi-GPU setup, assigns GPUs 0, 1, 2, ... N-1 round-robin.
    If more tasks than GPUs, cycles through available GPUs.
    
    Args:
        num_tasks: Number of tasks that need GPU assignment
        num_gpus: Number of GPUs available (auto-detect if None)
    
    Returns:
        List of GPU IDs, one per task. If not enough GPUs, cycles through available GPUs.
    """
    if num_gpus is None:
        num_gpus = detect_gpus()
    
    if num_gpus == 0:
        return [0] * num_tasks  # CPU fallback
    
    # Round-robin assignment if more tasks than GPUs
    return [i % num_gpus for i in range(num_tasks)]

def setup_gpu_environment(gpu_id: int):
    """Set CUDA_VISIBLE_DEVICES to isolate GPU for current process.
    
    This ensures each Ray task only sees its assigned GPU.
    Sets CUDA_VISIBLE_DEVICES=gpu_id so the task sees GPU 0 as its only GPU.
    
    Args:
        gpu_id: The actual GPU ID to use (will appear as GPU 0 to the process)
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

