#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <omp.h>
#include <mkl.h>
#include "procgrid.h"
#include "matrix.h"
#include "nystrom.h"

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Default values
    int p1 = 0, p2 = 0, p3 = 0;
    int n = 0, r = 0;
    std::string alg;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-p1" || arg == "--p1") {
            if (i + 1 < argc) {
                p1 = std::stoi(argv[++i]);
            }
        } else if (arg == "-p2" || arg == "--p2") {
            if (i + 1 < argc) {
                p2 = std::stoi(argv[++i]);
            }
        } else if (arg == "-p3" || arg == "--p3") {
            if (i + 1 < argc) {
                p3 = std::stoi(argv[++i]);
            }
        } else if (arg == "-n" || arg == "--n") {
            if (i + 1 < argc) {
                n = std::stoi(argv[++i]);
            }
        } else if (arg == "-r" || arg == "--r") {
            if (i + 1 < argc) {
                r = std::stoi(argv[++i]);
            }
        } else if (arg == "-alg" || arg == "--alg") { 
            if (i + 1 < argc) {
                alg = argv[++i]; // Store the string value
            }
        }
    }

    // Get MPI rank and size
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Check that the number of processes matches the expected grid size
    int p = nprocs;
    if (p != p1 * p2 * p3) {
    	std::cerr << p << " vs " << p1 << "*" << p2 << "*" << p3  << std::endl;
    	std::cerr << "Error: Number of processes does not match p1 * p2 * p3." << std::endl;
    	MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Print the parameters for verification (optional)
    if (myrank == 0) {
        printf("Nystrom approximation of %dx%d matrix to rank %d on %dx%dx%d grid\n", n, n, r, p1, p2, p3);
    }

    // Create the process grid
    ProcGrid grid(p1, p2, p3);
    //grid.printInfo();
    
    ParMat A(n, n, grid, 'A');
    A.generate();
    //A.printLocalMatrix();

    if(alg == "nystrom-1d-noredist-1d"){
        nystrom_1d_noredist_1d(A, r);
    }
    else if (alg == "nystrom-1d-redist-1d") {
        nystrom_1d_redist_1d(A, r);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
