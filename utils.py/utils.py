import torch

class GPUMonitor:
    """Utility for monitoring GPU memory usage"""
    
    def __init__(self, device):
        self.device = device
        
    def reset_peak_memory(self):
        """Reset peak memory stats"""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_current_memory(self):
        """Get current GPU memory allocated in bytes"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device)
        return 0
    
    def get_peak_memory(self):
        """Get peak GPU memory allocated in bytes"""
        if self.device.type == 'cuda':
            return torch.cuda.max_memory_allocated(self.device)
        return 0
    
    def synchronize(self):
        """Synchronize CUDA operations"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)


def get_device():
    """Get available device (cuda or cpu)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

