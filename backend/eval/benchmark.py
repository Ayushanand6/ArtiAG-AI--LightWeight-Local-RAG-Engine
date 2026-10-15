#!/usr/bin/env python3
"""
ArtiAG Search Benchmark Runner
Runs the high-performance Java benchmark comparing HNSW vs LinearScan across 1,000 and 10,000 vectors.
"""

import subprocess
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_benchmark():
    print("Running ArtiAG Vector Search Benchmark (1,000 and 10,000 vectors)...")
    cmd = ["mvn", "test", "-Dtest=BenchmarkTest"]
    result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    
    output = result.stdout
    start_marker = "=== ARTIAG VECTOR SEARCH BENCHMARK"
    if start_marker in output:
        idx = output.find(start_marker) - 19
        end_idx = output.find("==========================================================================================", idx + 100)
        print(output[idx:end_idx + 90])
    else:
        print(output)
        
    if result.returncode != 0:
        print("Benchmark failed!")
        sys.exit(1)

if __name__ == "__main__":
    run_benchmark()
