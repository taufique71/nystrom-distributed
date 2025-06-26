#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <omp.h>
#include <mkl.h>
#include "procgrid.h"
#include "matrix.h"

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Default values
    int p1 = 0, p2 = 0, p3 = 0;
    int n1 = 0, n2 = 0, n3 = 0;
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
        } else if (arg == "-n1" || arg == "--n1") {
            if (i + 1 < argc) {
                n1 = std::stoi(argv[++i]);
            }
        } else if (arg == "-n2" || arg == "--n2") {
            if (i + 1 < argc) {
                n2 = std::stoi(argv[++i]);
            }
        } else if (arg == "-n3" || arg == "--n3") {
            if (i + 1 < argc) {
                n3 = std::stoi(argv[++i]);
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
        printf("testing %dx%d with %dx%d on %dx%dx%d grid\n", n1, n2, n2, n3, p1, p2, p3);
    }

    // Create the process grid
    ProcGrid grid(p1, p2, p3);
    //grid.printInfo();
    
    ParMat A(n1, n2, grid, 'A');
    A.generate();
    //A.printLocalMatrix();
    
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(myrank == 0) std::cout << "---" << std::endl;

    ParMat B(n2, n3, grid, 'B');
    B.generate();
    //B.printLocalMatrix();

    if(alg == "matmul"){
        ParMat C = matmul(A, B);
    }
    else if (alg == "matmul1gen") {
        ParMat C = matmul1_gen(A, B, "xoroshiro");
    }
    else if (alg == "matmul1comm") {
        ParMat C = matmul1_comm(A, B, "xoroshiro");
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
