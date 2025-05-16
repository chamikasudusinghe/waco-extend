import os
import subprocess
import csv
import argparse
import re
import math
import numpy as np
import random
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def parse_output_file(filepath):
    best_schedule = None
    best_time = float('inf')
    fixed_csr_time = None

    sched_line_re = re.compile(r'Testing Candidate SuperSchedules\.\.\..*=\s*([\d\.]+)\s*ms')

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        match = sched_line_re.search(line)
        if match:
            time = float(match.group(1))
            if time < best_time:
                best_time = time
                best_schedule = line.split("Testing Candidate SuperSchedules...")[-1].strip()

        if line.startswith("FixedCSR"):
            fixed_csr_time = float(line.split(":")[-1].strip().replace(" ms", ""))

    if best_schedule is None or fixed_csr_time is None:
        raise ValueError("Missing WACO or FixedCSR time")

    return best_schedule, best_time, fixed_csr_time

def run_topk_evaluation(topk_dir, csr_dir, binary_path, mode, output_dir, max_matrices=None):
    results = []
    os.makedirs(output_dir, exist_ok=True)

    topk_files = sorted(f for f in os.listdir(topk_dir) if f.endswith(".txt"))
    
    if max_matrices is not None:
        topk_files = topk_files[:max_matrices]
        print("Testing only for ", len(topk_files), " matrices.")

    for fname in topk_files:
        matrix_name = fname[:-4]
        csr_path = os.path.join(csr_dir, matrix_name + ".csr")
        schedule_path = os.path.join(topk_dir, fname)
        output_path = os.path.join(output_dir, matrix_name + ".txt")

        if not os.path.exists(csr_path):
            print(f"[SKIP] Missing CSR file: {csr_path}")
            continue

        cmd = f"{binary_path} {csr_path} {schedule_path}"
        print(f"[RUN] {cmd}")

        with open(output_path, 'w') as out_file:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate()
            out_file.write(stdout)
            out_file.flush()

        try:
            schedule, waco_time, fixed_time = parse_output_file(output_path)
            results.append([matrix_name, mode, schedule, waco_time, fixed_time])
            print(f"[✓] {matrix_name}: {waco_time:.3f} ms (WACO), {fixed_time:.3f} ms (CSR)")
        except Exception as e:
            print(f"[WARN] Failed to parse {matrix_name}: {e}")

    return results

def save_results_to_csv(results, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['matrix_name', 'program_type', 'best_schedule', 'waco_time_ms', 'fixed_csr_time_ms'])
        writer.writerows(results)
    print(f"[✓] Saved results to {output_csv}")

def compute_geomean_speedup(results):
    speedups = []
    for row in results:
        _, _, _, waco_time, fixed_time = row
        if waco_time > 0:
            speedup = fixed_time / waco_time
            speedups.append(speedup)

    if not speedups:
        return None

    log_sum = sum(math.log(s) for s in speedups)
    geomean = math.exp(log_sum / len(speedups))
    return geomean

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['spmm', 'sddmm'], help="Program type to evaluate")
    parser.add_argument('--topk-dir', help="Directory containing top-k schedule .txt files")
    parser.add_argument('--csr-dir', help="Directory containing .csr matrix files")
    parser.add_argument('--binary', help="Path to the executable binary")
    parser.add_argument('--output-dir', help="Where to save raw execution outputs")
    parser.add_argument('--output-csv', help="Path to save final results CSV")
    parser.add_argument('--max-matrices', type=int, default=None, help="Maximum number of matrices to evaluate")
    args = parser.parse_args()

    ROOT = "/home/chamika2/waco-extend"

    mode = args.mode.lower()
    default_topk_dir = f"{ROOT}/WACO/COMMON/topk/{mode}"
    default_csr_dir = f"{ROOT}/dataset"
    default_binary = f"{ROOT}/code_generator/{mode}"
    default_output_dir = f"{ROOT}/WACO/COMMON/times/{mode}"
    default_output_csv = f"{ROOT}/WACO/COMMON/{mode}_results.csv"

    results = run_topk_evaluation(
        topk_dir=args.topk_dir or default_topk_dir,
        csr_dir=args.csr_dir or default_csr_dir,
        binary_path=args.binary or default_binary,
        mode=mode,
        output_dir=args.output_dir or default_output_dir,
        max_matrices=args.max_matrices,
    )

    save_results_to_csv(results, args.output_csv or default_output_csv)

    # geomean speedup
    geomean = compute_geomean_speedup(results)
    if geomean is not None:
        print(f"\n[✓] Geometric Mean Speedup over Fixed CSR: {geomean:.3f}×")
    else:
        print("\n[!] Geomean could not be computed — no valid results.")
