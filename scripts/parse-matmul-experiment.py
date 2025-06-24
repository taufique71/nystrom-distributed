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
    pattern = r'(\w+)\s+(\d+)x(\d+)\s+with\s+(\d+)x(\d+)\s+on\s+(\d+)x(\d+)x(\d+)\s+grid'
    match = re.search(pattern, content)
    if match:
        test_name = match.group(1)  # e.g., "testing"
        a_rows = int(match.group(2))  # Rows of matrix A
        a_cols = int(match.group(3))  # Columns of matrix A
        b_rows = int(match.group(4))  # Rows of matrix B
        b_cols = int(match.group(5))  # Columns of matrix B
        p1 = int(match.group(6))   # First dimension of the grid
        p2 = int(match.group(7))   # Second dimension of the grid
        p3 = int(match.group(8))   # Third dimension of the grid
    else:
        raise ValueError("Line format is incorrect.")

    # Extract timing information
    gather_a_time = re.search(r'Time to gather A:\s*([\d.]+) sec', content)
    gen_b_time = re.search(r'Time to generate B:\s*([\d.]+) sec', content)
    gather_b_time = re.search(r'Time to gather B:\s*([\d.]+) sec', content)
    local_multiply_time = re.search(r'Time for local multiply:\s*([\d.]+) sec', content)
    scatter_reduce_time = re.search(r'Time to scatter and reduce C:\s*([\d.]+) sec', content)

    return {
        'alg': alg,
        'impl': impl,
        'nnode': nnode,
        'nproc': nproc,
        'a_rows': a_rows,
        'a_cols': a_cols,
        'b_rows': b_rows,
        'b_cols': b_cols,
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'gather_a_time': float(gather_a_time.group(1)) if gather_a_time else 0,
        'gen_b_time': float(gen_b_time.group(1)) if gen_b_time else 0,
        'gather_b_time': float(gather_b_time.group(1)) if gather_b_time else 0,
        'local_multiply_time': float(local_multiply_time.group(1)) if local_multiply_time else 0,
        'scatter_reduce_time': float(scatter_reduce_time.group(1)) if scatter_reduce_time else 0,
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
    directory = '/pscratch/sd/t/taufique/nystrom'  # Change this to your directory
    output_file = 'matmul-results.csv'
    
    experiment_data = collect_experiment_data(directory)
    save_to_csv(experiment_data, output_file)
    # print(f"Data saved to {output_file}")
