#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <omp.h>
#include "procgrid.h"
#include "matrix.h"
#include "nystrom.h"

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Default values
    int matmul1p1 = 0, matmul1p2 = 0, matmul1p3 = 0;
    int matmul2p1 = 0, matmul2p2 = 0, matmul2p3 = 0;
    int n = 0, r = 0;
    std::string alg;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-matmul1p1" || arg == "--matmul1p1") {
            if (i + 1 < argc) {
                matmul1p1 = std::stoi(argv[++i]);
            }
        } else if (arg == "-matmul1p2" || arg == "--matmul1p2") {
            if (i + 1 < argc) {
                matmul1p2 = std::stoi(argv[++i]);
            }
        } else if (arg == "-matmul1p3" || arg == "--matmul1p3") {
            if (i + 1 < argc) {
                matmul1p3 = std::stoi(argv[++i]);
            }
        } else if (arg == "-matmul2p1" || arg == "--matmul2p1") {
            if (i + 1 < argc) {
                matmul2p1 = std::stoi(argv[++i]);
            }
        } else if (arg == "-matmul2p2" || arg == "--matmul2p2") {
            if (i + 1 < argc) {
                matmul2p2 = std::stoi(argv[++i]);
            }
        } else if (arg == "-matmul2p3" || arg == "--matmul2p3") {
            if (i + 1 < argc) {
                matmul2p3 = std::stoi(argv[++i]);
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

    //char MPI_version_string[MPI_MAX_LIBRARY_VERSION_STRING];
    //int MPI_version_length;
    //MPI_Get_library_version(MPI_version_string, &MPI_version_length);
    //if (myrank == 0) printf("%.*s\n", MPI_version_length, MPI_version_string);

    // Check that the number of processes matches the expected grid size
    int p = nprocs;
    if (p != matmul1p1 * matmul1p2 * matmul1p3) {
        std::cerr << p << " vs " << matmul1p1 << "*" << matmul1p2 << "*" << matmul1p3  << std::endl;
        std::cerr << "Error: Number of processes does not match matmul1p1 * matmul1p2 * matmul1p3." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (p != matmul2p1 * matmul2p2 * matmul2p3) {
        std::cerr << p << " vs " << matmul2p1 << "*" << matmul2p2 << "*" << matmul2p3  << std::endl;
        std::cerr << "Error: Number of processes does not match matmul2p1 * matmul2p2 * matmul2p3." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Print the parameters for verification
    if (myrank == 0) {
        printf("Nystrom approximation of %dx%d matrix to rank %d using %s\n", n, n, r, alg.c_str());
    }

    // Create the process grid
    ProcGrid grid1(matmul1p1, matmul1p2, matmul1p3);
    ProcGrid grid2(matmul2p1, matmul2p2, matmul2p3);
    //grid.printInfo();
    
    ParMat A(n, n, grid1, 'A');
    A.generate();
    //A.printLocalMatrix();

    if(alg == "nystrom-1d-noredist-1d"){
        ParMat Y(n, r, grid1, 'C');
        ParMat Z(r, r, grid1, 'B');
        nystrom_1d_noredist_1d(A, r, Y, Z);
    }
    else if (alg == "nystrom-1d-redist-1d") {
        ParMat Y(n, r, grid2, 'B');
        ParMat Z(r, r, grid2, 'C');
        nystrom_1d_redist_1d(A, r, Y, Z);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
