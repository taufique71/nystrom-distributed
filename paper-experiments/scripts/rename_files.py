import os
import re
import pandas as pd

if __name__ == "__main__":
    directory = '/pscratch/sd/t/taufique/nystrom/matmul_benchmarking'  # Change this to your directory
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        parts = filename.split('_')
        # matmul_python_8_64_4x4x4
        # matmul_cpp_perlmutter-cpu_1_4_64_10000_10000_10000_2x2x1

        alg = parts[0]
        impl = parts[1]
        system = parts[2]
        nnode = int(parts[3])
        nproc = int(parts[4])
        thread_per_proc = int(parts[5])
        m = int(parts[6])
        k = int(parts[7])
        n = int(parts[8])
        grid = parts[9]
        
        if alg == "matmul" :
            tot_n_thread = nnode * 128 * 2
            rep_n_thread = nproc * thread_per_proc
            if m != 50000:
                print(filename, "DO NOTHING")
            else:
                new_filename = alg + "_" + impl + \
                        "_" + system + "_" + str(nnode) + "_" + str(nproc) + "_" + str(thread_per_proc) + \
                        "_" + str(m) + "_" + str(k) + "_" + str(k) + "_" + grid
                new_file_path = os.path.join(directory, new_filename)
                os.rename(file_path, new_file_path)
                print(filename, "->", new_filename)
