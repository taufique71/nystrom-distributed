#!/bin/bash -l

PY=$HOME/Codes/nystrom-distributed/tests/matmul-test.py
BIN=$HOME/Codes/nystrom-distributed/build/c_matmul/matmul
#srun -N 1 -n 24 python $PY -p1 4 -p2 3 -p3 2 -n1 977 -n2 997 -n3 911
#srun -N 4 -n 128 python $PY -p1 128 -p2 1 -p3 1 -n1 1000 -n2 1000 -n3 10

#srun -N 1 -n 24 $BIN -p1 4 -p2 3 -p3 2 -n1 977 -n2 997 -n3 911
srun -N 1 -n 24 $BIN -p1 4 -p2 3 -p3 2 -n1 9 -n2 16 -n3 11
