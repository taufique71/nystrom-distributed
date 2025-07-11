import os
import re
import pandas as pd

def parse_experiment_file(file_path):
    filename = os.path.basename(file_path)

    parts = filename.split('_')
    alg = parts[0]
    impl = parts[1]
    nnode = int(parts[2])
    nproc = int(parts[3])

    with open(file_path, 'r') as file:    
        content = file.read()
    
    # Extract matrix dimensions (assuming they are in the format "MxN")
    dimension_match = re.search(r'(\d+)x(\d+)', content)
    # pattern = r'(\w+)\s+(\d+)x(\d+)\s+with\s+(\d+)x(\d+)\s+on\s+(\d+)x(\d+)x(\d+)\s+grid'
    # pattern = r"(\d+)x(\d+) matrix to rank (\d+) on (\d+)x(\d+)x(\d+) grid" 
    pattern = r"Nystrom approximation of (\d+)x(\d+) matrix to rank (\d+) using ([\w-]+)"
    match = re.search(pattern, content)
    if match:
        n = int(match.group(1))  # Rows of matrix A
        m = int(match.group(2))  # Columns of matrix A
        r = int(match.group(3))  # Target rank of Nystrom
    else:
        raise ValueError("Line format is incorrect.")

    # Extract timing information
    matmul1_grid = re.search(r'matmul1 in (\d+)x(\d+)x(\d+) grid', content)
    gen_omega_time = re.search(r'Time to generate Omega:\s*([\d.]+) sec', content)
    first_dgemm_time = re.search(r'Time for first dgemm:\s*([\d.]+) sec', content)
    matmul2_grid = re.search(r'matmul2 in (\d+)x(\d+)x(\d+) grid', content)
    redistrib_y_time = re.search(r'Time to redistribute Y:\s*([\d.]+) sec', content)
    second_dgemm_time = re.search(r'Time for second dgemm:\s*([\d.]+) sec', content)
    reduce_scatter_z_time = re.search(r'Time to scatter and reduce Z:\s*([\d.]+) sec', content)

    return {
        'alg': alg,
        'impl': impl,
        'nnode': nnode,
        'nproc': nproc,
        'n': n,
        'm': m,
        'r': r,
        'matmul1p1': int(matmul1_grid.group(1)),
        'matmul1p2': int(matmul1_grid.group(2)),
        'matmul1p3': int(matmul1_grid.group(3)),
        'matmul2p1': int(matmul2_grid.group(1)),
        'matmul2p2': int(matmul2_grid.group(2)),
        'matmul2p3': int(matmul2_grid.group(3)),
        'gen_omega_time': float(gen_omega_time.group(1)) if gen_omega_time else 0,
        'first_dgemm_time': float(first_dgemm_time.group(1)) if first_dgemm_time else 0,
        'redistrib_y_time': float(redistrib_y_time.group(1)) if redistrib_y_time else 0,
        'second_dgemm_time': float(second_dgemm_time.group(1)) if second_dgemm_time else 0,
        'reduce_scatter_z_time': float(reduce_scatter_z_time.group(1)) if reduce_scatter_z_time else 0,
    }

def collect_experiment_data(directory):
    data = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        experiment_data = parse_experiment_file(file_path)
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
