import argparse
from benchmark.core.config import Config
from benchmark.core.benchmark_runner import BenchmarkRunner
from benchmark.core.ray_runner import RayBenchmarkRunner
from benchmark.core.nsight_profiler import NsightProfiler
from benchmark.core.prometheus_metrics import PrometheusMetricsExporter
from benchmark.utils.device import get_device
from benchmark.utils.output import ResultsWriter
from benchmark.utils.factories import get_compiler, get_model

def main():
    """Entry point that loads config, runs all requested cases, and saves CSV.

    Reads `config.yaml`, builds model wrappers, iterates over models, compilers and
    batch sizes, and writes a single results file under the configured output
    directory. Supports both sequential and distributed (Ray) execution modes.
    """
    parser = argparse.ArgumentParser(description='ML Compiler Benchmark Framework')
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Enable distributed execution with Ray (multi-GPU parallel)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    args = parser.parse_args()
    
    cfg = Config.from_yaml(args.config)
    
    use_ray = args.distributed or cfg.ray.enabled
    
    print("="*70)
    print("ML COMPILER BENCHMARK FRAMEWORK")
    print("="*70)
    print(f"Execution mode: {'Distributed (Ray)' if use_ray else 'Sequential'}")
    print(f"Models: {', '.join([m.name for m in cfg.models])}")
    print(f"Compilers: {', '.join(cfg.compilers)}")
    print(f"Warmup iterations: {cfg.benchmark.warmup_iterations}")
    print(f"Measured iterations: {cfg.benchmark.measured_iterations}")
    print("="*70)
    
    # Initialize Prometheus metrics exporter if enabled
    prometheus_exporter = None
    if cfg.monitoring.prometheus.enabled:
        prometheus_exporter = PrometheusMetricsExporter(
            port=cfg.monitoring.prometheus.port,
            enabled=cfg.monitoring.prometheus.enabled
        )
    
    if use_ray:
        try:
            ray_runner = RayBenchmarkRunner(
                num_gpus=cfg.ray.num_gpus,
                num_cpus=cfg.ray.num_cpus,
                head_address=cfg.ray.head_address,
                resources_per_task=cfg.ray.resources_per_task
            )
            
            models_config = []
            for model_cfg in cfg.models:
                model_dict = {
                    'name': model_cfg.name,
                    'batch_sizes': model_cfg.batch_sizes
                }
                if model_cfg.input_shape:
                    model_dict['input_shape'] = model_cfg.input_shape
                if model_cfg.max_length:
                    model_dict['max_length'] = model_cfg.max_length
                models_config.append(model_dict)
            
            # Prepare profiling config for Ray tasks
            profiling_config = None
            if cfg.profiling.enabled:
                profiling_config = {
                    'enabled': cfg.profiling.enabled,
                    'output_dir': cfg.profiling.output_dir,
                    'profile_iterations': cfg.profiling.profile_iterations
                }
            
            # Prepare Prometheus config for Ray tasks
            prometheus_config = None
            if cfg.monitoring.prometheus.enabled:
                prometheus_config = {
                    'enabled': cfg.monitoring.prometheus.enabled,
                    'port': cfg.monitoring.prometheus.port
                }
            
            combined_results = ray_runner.run_distributed_benchmarks(
                models_config=models_config,
                compiler_names=cfg.compilers,
                warmup_iters=cfg.benchmark.warmup_iterations,
                measured_iters=cfg.benchmark.measured_iterations,
                profiling_config=profiling_config,
                prometheus_config=prometheus_config
            )
            
            ray_runner.shutdown()
            
        except ImportError:
            print("Error: Ray is not installed. Install with: pip install ray")
            print("Falling back to sequential execution...")
            use_ray = False
        except Exception as e:
            print(f"Error initializing Ray: {e}")
            print("Falling back to sequential execution...")
            use_ray = False
    
    if not use_ray:
        device = get_device()
        
        # Create Nsight profiler if enabled
        nsight_profiler = None
        if cfg.profiling.enabled:
            nsight_profiler = NsightProfiler(
                output_dir=cfg.profiling.output_dir,
                enabled=cfg.profiling.enabled,
                profile_iterations=cfg.profiling.profile_iterations
            )
        
        runner = BenchmarkRunner(
            device=device,
            warmup_iters=cfg.benchmark.warmup_iterations,
            measured_iters=cfg.benchmark.measured_iterations,
            nsight_profiler=nsight_profiler,
            prometheus_exporter=prometheus_exporter
        )
        
        combined_results = []
        
        for model_config in cfg.models:
            print(f"\n{'#'*70}")
            print(f"Processing model: {model_config.name}")
            print(f"{'#'*70}")
            
            model_wrapper = get_model(model_config)
            
            for compiler_name in cfg.compilers:
                compiler = get_compiler(compiler_name)
                
                for batch_size in model_config.batch_sizes:
                    try:
                        run_stats = runner.run_benchmark(model_wrapper, compiler, batch_size)
                        combined_results.append(run_stats)
                    except Exception as e:
                        print(f"\n✗ Error running benchmark for {model_config.name} | {compiler_name} | batch_size={batch_size}: {e}")
                        print("  Continuing with remaining benchmarks...")
                        # Continue with other benchmarks instead of crashing
    
    output_path = f"{cfg.output.save_path}/benchmark_results.csv"
    ResultsWriter.write_csv(combined_results, output_path)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print(f"Results saved to: {output_path}")
    print(f"Total benchmarks completed: {len(combined_results)}")
    print("="*70)

if __name__ == "__main__":
    main()
