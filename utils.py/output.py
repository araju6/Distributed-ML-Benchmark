import csv
import os
from typing import List
from ..core.metrics import BenchmarkMetrics

class ResultsWriter:
    
    @staticmethod
    def write_csv(results: List[BenchmarkMetrics], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not results:
            print("No results to write!")
            return
        
        fieldnames = list(results[0].to_dict().keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())
        
        print(f"\n✓ Results saved to: {output_path}")

