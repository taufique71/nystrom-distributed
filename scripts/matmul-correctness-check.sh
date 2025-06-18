#!/bin/bash -l

module load python

PY=$HOME/Codes/nystrom-distributed/tests/matmul-correctness-test.py
srun -N 1 -n 128 python $PY -p1 8 -p2 4 -p3 4 -n1 971 -n2 997 -n3 983
