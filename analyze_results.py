import csv
import sys
import os

def analyze_results(csv_path="results/benchmark_results.csv"):
    """Read a CSV of runs and print a small summary + comparison.

    Assumes the CSV was produced by this repo (matching headers). This is just
    a quick console view to eyeball differences; for real analysis you'd pull
    it into a notebook.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        return
    
    by_case = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model']
            compiler = row['compiler']
            batch_size = int(row['batch_size'])
            key = (model, compiler, batch_size)
            by_case[key] = {
                'latency_mean_ms': float(row['latency_mean_ms']),
                'throughput_samples_per_sec': float(row['throughput_samples_per_sec']),
                'compile_time_sec': row.get('compile_time_sec', 'N/A')
            }
    
    print("="*70)
    print("BENCHMARK RESULTS ANALYSIS")
    print("="*70)
    print()
    
    # Show results by model, compiler, and batch size
    models = sorted(set(m for m, _, _ in by_case.keys()))
    compilers = sorted(set(c for _, c, _ in by_case.keys()))
    batch_sizes = sorted(set(b for _, _, b in by_case.keys()))
    
    for model in models:
        print(f"\n{'#'*70}")
        print(f"Model: {model}")
        print(f"{'#'*70}")
        
        for compiler in compilers:
            print(f"\nCompiler: {compiler}")
            print("-" * 70)
            for batch_size in batch_sizes:
                key = (model, compiler, batch_size)
                if key in by_case:
                    stat_row = by_case[key]
                    print(f"  Batch Size {batch_size}:")
                    print(f"    Latency:     {stat_row['latency_mean_ms']:.2f} ms")
                    print(f"    Throughput:  {stat_row['throughput_samples_per_sec']:.2f} samples/sec")
                    if stat_row['compile_time_sec'] != 'N/A':
                        compile_time = float(stat_row['compile_time_sec'])
                        print(f"    Compile:     {compile_time:.2f} s")
                    print()
    
    # Compare compilers (compare eager vs all others if available)
    print("="*70)
    eager_compiler = 'pytorch_eager'
    other_compilers = [c for c in compilers if c != eager_compiler]
    
    if eager_compiler in compilers and other_compilers:
        print(f"COMPARISON: {eager_compiler} vs Other Compilers")
        print("="*70)
        print()
        
        for model in models:
            print(f"\nModel: {model}")
            print("-" * 70)
            for other_compiler in other_compilers:
                for batch_size in batch_sizes:
                    eager_key = (model, eager_compiler, batch_size)
                    other_key = (model, other_compiler, batch_size)
                    
                    if eager_key in by_case and other_key in by_case:
                        eager = by_case[eager_key]
                        other = by_case[other_key]
                        
                        latency_speedup = eager['latency_mean_ms'] / other['latency_mean_ms']
                        throughput_speedup = other['throughput_samples_per_sec'] / eager['throughput_samples_per_sec']
                        
                        print(f"  {eager_compiler} vs {other_compiler} (Batch {batch_size}):")
                        print(f"    Latency:    {eager['latency_mean_ms']:.2f} ms → {other['latency_mean_ms']:.2f} ms")
                        print(f"    Speedup:    {latency_speedup:.2f}x faster ({((latency_speedup-1)*100):.1f}% improvement)")
                        print(f"    Throughput: {eager['throughput_samples_per_sec']:.2f} → {other['throughput_samples_per_sec']:.2f} samples/sec")
                        print(f"    Improvement: {throughput_speedup:.2f}x ({((throughput_speedup-1)*100):.1f}% increase)")
                        
                        if other['compile_time_sec'] != 'N/A':
                            compile_time = float(other['compile_time_sec'])
                            print(f"    Compile Cost: {compile_time:.2f} s (one-time, amortized over runs)")
                        print()
    
    print("="*70)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.csv"
    analyze_results(csv_path)
