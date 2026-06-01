#include <mpi.h>
#include <iostream>
#include <string>
#include "procgrid.h"
#include "matrix.h"
#include "nystrom.h"

static void print_usage() {
    std::cout <<
"Usage: nystrom -alg <algorithm> -n <n> -r <r>\n"
"               [-afile <input.bin>] [-yfile <Y.bin>] [-zfile <Z.bin>]\n"
"\n"
"Compute Nystrom low-rank approximation factors Y, Z of an n x n symmetric\n"
"matrix A to target rank r. All matrices are binary, column-major, double\n"
"precision. Process distribution is 1D, derived from MPI nprocs.\n"
"\n"
"Required arguments:\n"
"  -alg <name>           Algorithm variant. One of:\n"
"                          nystrom-1d-noredist-1d\n"
"                          nystrom-1d-redist-1d\n"
"  -n <int>              Matrix dimension (matrix is n x n)\n"
"  -r <int>              Target rank (r <= n)\n"
"\n"
"Optional arguments:\n"
"  -afile <path>         Input matrix A. If omitted, A is generated locally\n"
"                        as a deterministic test pattern (benchmarking only).\n"
"  -yfile <path>         Output file for Y factor. If omitted, Y is computed\n"
"                        but not written to disk.\n"
"  -zfile <path>         Output file for Z factor. If omitted, Z is computed\n"
"                        but not written to disk.\n"
"  -h, --help            Print this message and exit.\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Defaults
    int n = 0, r = 0;
    std::string alg;
    std::string afile = "NONE";
    std::string yfile = "NONE";
    std::string zfile = "NONE";

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            if (myrank == 0) print_usage();
            MPI_Finalize();
            return 0;
        }
        else if ((arg == "-n"     || arg == "--n")     && i + 1 < argc) n     = std::stoi(argv[++i]);
        else if ((arg == "-r"     || arg == "--r")     && i + 1 < argc) r     = std::stoi(argv[++i]);
        else if ((arg == "-alg"   || arg == "--alg")   && i + 1 < argc) alg   = argv[++i];
        else if ((arg == "-afile" || arg == "--afile") && i + 1 < argc) afile = argv[++i];
        else if ((arg == "-yfile" || arg == "--yfile") && i + 1 < argc) yfile = argv[++i];
        else if ((arg == "-zfile" || arg == "--zfile") && i + 1 < argc) zfile = argv[++i];
        else {
            if (myrank == 0) {
                std::cerr << "Error: unknown or malformed argument '" << arg << "'\n\n";
                print_usage();
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Validation
    auto fail = [&](const std::string& msg) {
        if (myrank == 0) {
            std::cerr << "Error: " << msg << "\n\n";
            print_usage();
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    };

    if (alg.empty())                                       fail("missing required -alg");
    if (n <= 0)                                            fail("missing or invalid -n (must be > 0)");
    if (r <= 0)                                            fail("missing or invalid -r (must be > 0)");
    if (r > n)                                             fail("invalid -r (must be <= n)");
    if (alg != "nystrom-1d-noredist-1d" &&
        alg != "nystrom-1d-redist-1d")                     fail("unknown algorithm '" + alg + "'");

    if (myrank == 0) {
        printf("Nystrom approximation of %dx%d matrix to rank %d using %s\n",
               n, n, r, alg.c_str());
    }

    if (alg == "nystrom-1d-noredist-1d") {
        ProcGrid grid1(nprocs, 1, 1);
        ProcGrid grid2(nprocs, 1, 1);

        ParMat A(n, n, grid1, 'A');
        if (afile == "NONE") A.generate();
        else A.parallelReadBinary(afile, MPI_COMM_WORLD);

        ParMat Y(n, r, grid1, 'C');
        ParMat Z(r, r, grid1, 'B');
        nystrom_1d_noredist_1d(A, r, Y, Z);

        if (yfile != "NONE") Y.parallelWriteBinary(yfile, MPI_COMM_WORLD);
        if (zfile != "NONE") Z.parallelWriteBinary(zfile, MPI_COMM_WORLD);
    }
    else if (alg == "nystrom-1d-redist-1d") {
        ProcGrid grid1(nprocs, 1, 1);
        ProcGrid grid2(1, 1, nprocs);

        ParMat A(n, n, grid1, 'A');
        if (afile == "NONE") A.generate();
        else A.parallelReadBinary(afile, MPI_COMM_WORLD);

        ParMat Y(n, r, grid2, 'B');
        ParMat Z(r, r, grid2, 'C');
        nystrom_1d_redist_1d(A, r, Y, Z);

        if (yfile != "NONE") Y.parallelWriteBinary(yfile, MPI_COMM_WORLD);
        if (zfile != "NONE") Z.parallelWriteBinary(zfile, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
