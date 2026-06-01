import sys
import argparse
from mpi4py import MPI

from communicator import ProcGrid
from matrix import ParMat, matmul, matmul1_gen, matmul1_comm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="matmul",
        description=(
            "Compute the parallel matrix product C = A * B on a p1 x p2 x p3\n"
            "MPI process grid. A is n1 x n2, B is n2 x n3. All matrices are\n"
            "binary, column-major, double precision."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-alg", required=True,
                        choices=["matmul", "matmul1gen", "matmul1comm"],
                        help="Algorithm variant")
    parser.add_argument("-n1", type=int, required=True, help="Rows of A")
    parser.add_argument("-n2", type=int, required=True, help="Columns of A / rows of B")
    parser.add_argument("-n3", type=int, required=True, help="Columns of B")
    parser.add_argument("-p1", type=int, required=True, help="Process grid rows")
    parser.add_argument("-p2", type=int, required=True, help="Process grid columns")
    parser.add_argument("-p3", type=int, required=True, help="Process grid fibers")
    parser.add_argument("-file", default="NONE",
                        help="Input matrix A file. If omitted, A is generated locally.")
    args = parser.parse_args()

    myrank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()

    if args.p1 * args.p2 * args.p3 != nprocs:
        if myrank == 0:
            print(f"Error: p1*p2*p3 ({args.p1}*{args.p2}*{args.p3}) does not match nprocs ({nprocs})",
                  file=sys.stderr)
        MPI.COMM_WORLD.Abort(1)

    if myrank == 0:
        print(f"testing {args.n1}x{args.n2} with {args.n2}x{args.n3} on {args.p1}x{args.p2}x{args.p3} grid")

    grid = ProcGrid(args.p1, args.p2, args.p3)
    A = ParMat(args.n1, args.n2, grid, 'A')
    if args.file == "NONE":
        A.generate()
    else:
        A.parallelReadBinary(args.file, MPI.COMM_WORLD)

    B = ParMat(args.n2, args.n3, grid, 'B')
    B.generate()

    if args.alg == "matmul":
        C = matmul(A, B)
    elif args.alg == "matmul1gen":
        C = matmul1_gen(A, B, "xoroshiro")
    elif args.alg == "matmul1comm":
        C = matmul1_comm(A, B, "xoroshiro")
