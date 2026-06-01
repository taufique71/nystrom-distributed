import os
import sys
import argparse
import numpy as np
import math
import random

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-afile", "--afile", type=str, help="A file name")
    parser.add_argument("-yfile", "--yfile", type=str, help="Y file name")
    parser.add_argument("-zfile", "--zfile", type=str, help="Z file name")
    args = parser.parse_args()
    n = 50000
    d = 3072
    # r = 5000
    r = 2500
    kernel = "linear"
    # kernel = "rbf"
    system = "cpu"
    print(n, d, r, kernel, system)
    print("---")
    dfile = "/pscratch/sd/t/taufique/nystrom/dataset/cifar10.bin"
    afile = "/pscratch/sd/t/taufique/nystrom/dataset/cifar10-rbf-sig1.bin"
    # afile = "/pscratch/sd/t/taufique/nystrom/dataset/cifar10-"+kernel+".bin"
    # yfile = "/pscratch/sd/t/taufique/nystrom/dataset/cifar10-"+kernel+"-" +str(r)+"-y.bin"
    # zfile = "/pscratch/sd/t/taufique/nystrom/dataset/cifar10-"+kernel+"-" +str(r)+"-z.bin"
    # if system == "cpu":
        # yfile = "/pscratch/sd/t/taufique/nystrom/dataset/cifar10-"+kernel+"-"+system+"-"+str(r)+"-y.bin"
        # zfile = "/pscratch/sd/t/taufique/nystrom/dataset/cifar10-"+kernel+"-"+system+"-"+str(r)+"-z.bin"

    D = np.fromfile(dfile, dtype=np.float64).reshape((n, d), order='F')
    A = np.fromfile(afile, dtype=np.float64).reshape((n, n), order='F')
    # Y = np.fromfile(yfile, dtype=np.float64).reshape((n, r), order='F')
    # Z = np.fromfile(zfile, dtype=np.float64).reshape((r, r), order='F')

    # D_norm = np.linalg.norm(D, ord='fro')
    # print(D_norm, D_norm/math.sqrt(n))
    # svdvals = np.linalg.svdvals(D)
    # print("svdvals", svdvals)
    # print("500th svdval", svdvals[499])

    # Z_pinv = np.linalg.pinv(Z, hermitian=True, rtol=1e-12)
    # A_aprx = Y @ Z_pinv @ Y.T
    # A_diff = A - A_aprx
    # diff_norm = np.linalg.norm(A_diff, ord='fro')
    A_norm = np.linalg.norm(A, ord='fro')
    # print("Absolute err", diff_norm)
    # print("Relative err", diff_norm/A_norm)
    
    rng = np.random.default_rng()
    Onp = rng.uniform(low=0.0, high=1.0, size=(n, r)).astype(np.float64)
    Ynp = A @ Onp
    Znp = Onp.T @ Ynp
    Znp_pinv = np.linalg.pinv(Znp, hermitian=True, rtol=1e-12)
    Anp_aprx = Ynp @ Znp_pinv @ Ynp.T
    Anp_diff = A - Anp_aprx
    diff_norm_np = np.linalg.norm(Anp_diff, ord='fro')
    print("Absolute err", diff_norm_np)
    print("Relative err", diff_norm_np/A_norm)

