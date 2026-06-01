import os
import re
import pandas as pd
from datetime import datetime


def parse_experiment_file(file_path):
    filename = os.path.basename(file_path)
    parts = re.split(r"[_.x]+", filename)

    alg = parts[0]
    impl = parts[1]
    system = parts[2]
    nnode = int(parts[3])
    nproc = int(parts[4])
    thread_per_proc = int(parts[5])
    m = int(parts[6])
    k = int(parts[7])
    n = int(parts[8])
    p = int(parts[9])
    q = int(parts[10])
    # parts[11] is always "1" (the dummy third grid dim for SLATE 2D)
    try_num = int(parts[12])

    with open(file_path, 'r') as file:
        content = file.read()

    # Header sanity check (matches the format printed by slate-matmul.cpp)
    header_pattern = r'testing (\d+)x(\d+) with (\d+)x(\d+) on (\d+)x(\d+)x(\d+) grid'
    if not re.search(header_pattern, content):
        return None

    nb_match = re.search(r'SLATE tile size nb = (\d+)', content)
    time_match = re.search(r'Time for SLATE gemm:\s*([\d.]+) sec', content)
    perf_match = re.search(r'Performance:\s*([\d.]+) GF/s', content)

    if not time_match:
        # Skip files without a timing line (failed runs)
        return None

    timestamp = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

    return {
        'alg': alg,
        'impl': impl,
        'system': system,
        'nnode': nnode,
        'nproc': nproc,
        'thread_per_proc': thread_per_proc,
        'm': m,
        'k': k,
        'n': n,
        'p': p,
        'q': q,
        'try': try_num,
        'nb': int(nb_match.group(1)) if nb_match else 0,
        'gemm_time': float(time_match.group(1)),
        'gflops': float(perf_match.group(1)) if perf_match else 0,
        'timestamp': timestamp,
    }


def collect_experiment_data(directory):
    data = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        experiment_data = parse_experiment_file(file_path)
        if experiment_data is not None:
            data.append(experiment_data)
        else:
            print("[.]", filename)
    return data


def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    directory = '/pscratch/sd/t/taufique/nystrom/slate-matmul_benchmarking'
    output_file = 'slate-results.csv'

    experiment_data = collect_experiment_data(directory)
    save_to_csv(experiment_data, output_file)
