#!/bin/bash -l

PY=/global/u1/t/taufique/Codes/nystrom-distributed/tests/matmul-test.py
srun -N 1 -n 24 python $PY -p1 4 -p2 3 -p3 2 -n1 977 -n2 997 -n3 911
