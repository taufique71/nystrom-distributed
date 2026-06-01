import os
import re
import pandas as pd

def parse_experiment_file(file_path):
    filename = os.path.basename(file_path)
    
    # nystrom-1d-noredist-1d_cpp_perlmutter-gpu_32_128_32_50000_5000_128x1x1_128x1x1.10
    parts = re.split(r"[_.x]+", filename)
    # print(parts)
    # parts = filename.split('_.')
    alg = parts[0]
    impl = parts[1]
    system = parts[2]
    nnode = int(parts[3])
    nproc = int(parts[4])
    nthread = int(parts[5])
    n = int(parts[6])
    r = int(parts[7])
    matmul1p1 = int(parts[8])
    matmul1p2 = int(parts[9])
    matmul1p3 = int(parts[10])
    matmul2p1 = int(parts[11])
    matmul2p2 = int(parts[12])
    matmul2p3 = int(parts[13])

    with open(file_path, 'r') as file:    
        content = file.read()
    
    # Extract matrix dimensions (assuming they are in the format "MxN")
    dimension_match = re.search(r'(\d+)x(\d+)', content)
    pattern = r"Nystrom approximation of (\d+)x(\d+) matrix to rank (\d+) using ([\w-]+)"
    match = re.search(pattern, content)
    if match:
        assert n == int(match.group(1)) # Rows of matrix A
        assert n == int(match.group(2)) # Columns of matrix A
        assert r == int(match.group(3)) # Target rank of Nystrom
    else:
        # raise ValueError("Line format is incorrect.")
        return None

    # Extract timing information
    total_time = re.search(r'Total time:\s*([\d.]+) sec', content)
    gen_omega_time = re.search(r'Time to generate Omega:\s*([\d.]+) sec', content)
    dgemm1_time = re.search(r'Time for first dgemm:\s*([\d.]+) sec', content)
    dgemm2_time = re.search(r'Time for second dgemm:\s*([\d.]+) sec', content)
    all2all_time = re.search(r'Time for all2all:\s*([\d.]+) sec', content)
    reduce_scatter_time = re.search(r'Time for reduce-scatter:\s*([\d.]+) sec', content)
    reduce_scatter_1_time = re.search(r'Time for first reduce-scatter:\s*([\d.]+) sec', content)
    reduce_scatter_2_time = re.search(r'Time for second reduce-scatter:\s*([\d.]+) sec', content)
    unpack_time = re.search(r'Time to unpack:\s*([\d.]+) sec', content)
    pack_time = re.search(r'Time to pack for reduce-scatter:\s*([\d.]+) sec', content)

    return {
        'alg': alg,
        'impl': impl,
        'system': system,
        'nnode': nnode,
        'nproc': nproc,
        'nthread': nthread,
        'n': n,
        'r': r,
        'matmul1p1': matmul1p1,
        'matmul1p2': matmul1p2,
        'matmul1p3': matmul1p3,
        'matmul2p1': matmul2p1,
        'matmul2p2': matmul2p2,
        'matmul2p3': matmul2p3,
        'total_time': float(total_time.group(1)) if total_time else 0,
        'gen_omega_time': float(gen_omega_time.group(1)) if gen_omega_time else 0,
        'dgemm1_time': float(dgemm1_time.group(1)) if dgemm1_time else 0,
        'dgemm2_time': float(dgemm2_time.group(1)) if dgemm2_time else 0,
        'all2all_time': float(all2all_time.group(1)) if all2all_time else 0,
        'reduce_scatter_time': float(reduce_scatter_time.group(1)) if reduce_scatter_time else 0,
        'reduce_scatter_1_time': float(reduce_scatter_1_time.group(1)) if reduce_scatter_1_time else 0,
        'reduce_scatter_2_time': float(reduce_scatter_2_time.group(1)) if reduce_scatter_2_time else 0,
        'unpack_time': float(unpack_time.group(1)) if unpack_time else 0,
        'pack_time': float(pack_time.group(1)) if pack_time else 0,
    }

def collect_experiment_data(directory):
    data = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        experiment_data = parse_experiment_file(file_path)
        if experiment_data is None:
            print("[.]", file_path)
            # os.remove(file_path)
            # pass
        else:
            data.append(experiment_data)
    return data

def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    directory = '/pscratch/sd/t/taufique/nystrom/nystrom_benchmarking'  # Change this to your directory
    output_file = 'nystrom-results.csv'
    
    experiment_data = collect_experiment_data(directory)
    save_to_csv(experiment_data, output_file)
    # print(f"Data saved to {output_file}")
