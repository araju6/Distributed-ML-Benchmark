import os
import subprocess
import shutil
from typing import Optional
from pathlib import Path

class NsightProfiler:
    """Wrapper for NVIDIA Nsight Systems profiling tool.
    
    Nsight Systems provides system-wide performance analysis including
    CUDA kernel execution, memory transfers, and CPU-GPU synchronization.
    
    Designed for containerized environments where nsys should be in PATH
    or installed in standard container locations.
    
    Usage:
        profiler = NsightProfiler(enabled=True, output_dir="results/profiles")
        profile_path = profiler.run_profiling_subprocess(...)
    """
    
    def __init__(self, output_dir: str = "results/profiles", enabled: bool = True, profile_iterations: int = 10):
        """Initialize Nsight Systems profiler.
        
        Args:
            output_dir: Directory to save profiling reports
            enabled: Whether profiling is enabled
            profile_iterations: Number of iterations to profile
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.profile_iterations = profile_iterations
        self.nsys_path = self._find_nsys()
        
        if enabled and self.nsys_path is None:
            print("Warning: Nsight Systems (nsys) not found. Profiling will be disabled.")
            print("Install Nsight Systems from: https://developer.nvidia.com/nsight-systems")
            self.enabled = False
    
    def _find_nsys(self) -> Optional[str]:
        """Find nsys executable in PATH or standard container locations."""
        # First check PATH (standard for containerized environments)
        nsys_path = shutil.which("nsys")
        if nsys_path:
            return nsys_path
        
        # Standard container installation path
        container_path = "/opt/nvidia/nsight-systems/bin/nsys"
        if os.path.exists(container_path) and os.access(container_path, os.X_OK):
            return container_path
        
        return None
    
    def is_available(self) -> bool:
        """Check if Nsight Systems is available."""
        return self.nsys_path is not None and self.enabled
    
    def run_profiling_subprocess(
        self,
        model_name: str,
        compiler_name: str,
        batch_size: int,
        model_config: dict,
        num_iterations: int = 10
    ) -> Optional[str]:
        """Run profiling in a subprocess with nsys.
        
        Args:
            model_name: Name of the model
            compiler_name: Name of the compiler
            batch_size: Batch size
            model_config: Model configuration dict (input_shape or max_length)
            num_iterations: Number of iterations to profile
        
        Returns:
            Path to profile file if successful, None otherwise
        """
        if not self.is_available():
            return None
        
        profile_name = f"{model_name}_{compiler_name}_bs{batch_size}"
        profile_file = self.get_profile_path(profile_name)
        
        # Build command to run profile_benchmark.py with nsys
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "scripts",
            "profile_benchmark.py"
        )
        
        if not os.path.exists(script_path):
            print(f"Warning: Profile script not found: {script_path}")
            return None
        
        # Build arguments for profile script
        profile_args = [
            "--model", model_name,
            "--compiler", compiler_name,
            "--batch-size", str(batch_size),
            "--iterations", str(num_iterations)
        ]
        
        if "input_shape" in model_config:
            profile_args.extend(["--input-shape"] + [str(x) for x in model_config["input_shape"]])
        if "max_length" in model_config:
            profile_args.extend(["--max-length", str(model_config["max_length"])])
        
        # Build nsys command
        nsys_cmd = [
            self.nsys_path,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--output", str(profile_file),
            "--force-overwrite=true",
            "--cuda-memory-usage=true",
            "--gpu-metrics-device=all",
            "python", script_path
        ] + profile_args
        
        try:
            print(f"Running Nsight Systems profiling...")
            result = subprocess.run(
                nsys_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if profile_file.exists():
                print(f"✓ Nsight profile saved: {profile_file}")
                return str(profile_file)
            else:
                print(f"Warning: Profile file not created: {profile_file}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error running Nsight Systems: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            return None
        except Exception as e:
            print(f"Unexpected error during profiling: {e}")
            return None
    
    def should_profile_iterations(self, iteration_num: int, total_iterations: int, profile_sample_size: int = 10) -> bool:
        """Determine if current iteration should be profiled.
        
        We profile a sample of iterations (e.g., first 10) to avoid overhead.
        
        Args:
            iteration_num: Current iteration number (0-indexed)
            total_iterations: Total number of iterations
            profile_sample_size: Number of iterations to profile
        
        Returns:
            True if this iteration should be profiled
        """
        if not self.is_available():
            return False
        
        # Profile first N iterations
        return iteration_num < profile_sample_size
    
    def get_profile_path(self, profile_name: str) -> Path:
        """Get the expected path for a profile file.
        
        Args:
            profile_name: Name of the profile (without extension)
        
        Returns:
            Path object for the profile file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{profile_name}.nsys-rep"

