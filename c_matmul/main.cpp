#include <mpi.h>
#include <iostream>
#include <vector>

class ProcGrid {
public:
    ProcGrid(int nProcRow, int nProcCol, int nProcFib) 
        : nProcRow(nProcRow), nProcCol(nProcCol), nProcFib(nProcFib) {
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

        // Calculate ranks in the 3D grid
		rowRank = myrank / (nProcCol * nProcFib);
		colRank = (myrank / nProcFib) % nProcCol;
		fibRank = myrank % nProcFib;

		std::vector<std::vector<std::vector<int>>> rowGroupRanks;
		std::vector<std::vector<std::vector<int>>> colGroupRanks;
		std::vector<std::vector<std::vector<int>>> fibGroupRanks;

		// Initialize rowGroupRanks
        rowGroupRanks.resize(nProcRow);
        for (int i = 0; i < nProcRow; ++i) {
            rowGroupRanks[i].resize(nProcFib);
        }

        // Initialize colGroupRanks
        colGroupRanks.resize(nProcCol);
        for (int i = 0; i < nProcCol; ++i) {
            colGroupRanks[i].resize(nProcFib);
        }

        // Initialize fibGroupRanks
        fibGroupRanks.resize(nProcRow);
        for (int i = 0; i < nProcRow; ++i) {
            fibGroupRanks[i].resize(nProcCol);
        }

        // Populate the group ranks
        for (int i = 0; i < nprocs; ++i) {
            int a = i / (nProcCol * nProcFib); // Row index
            int b = (i / nProcFib) % nProcCol; // Column index
            int c = i % nProcFib;               // Fiber index

            rowGroupRanks[a][c].push_back(i);
            colGroupRanks[b][c].push_back(i);
            fibGroupRanks[a][b].push_back(i);
        }

		MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

		// Create row world
		std::vector<int> rowRanks = rowGroupRanks[rowRank][fibRank];
		MPI_Group_incl(worldGroup, rowRanks.size(), rowRanks.data(), &rowGroup);
		MPI_Comm_create(MPI_COMM_WORLD, rowGroup, &rowWorld);
        MPI_Comm_rank(rowWorld, &rankInRowWorld);

		// Create column world
		std::vector<int> colRanks = colGroupRanks[colRank][fibRank];
		MPI_Group_incl(worldGroup, colRanks.size(), colRanks.data(), &colGroup);
		MPI_Comm_create(MPI_COMM_WORLD, colGroup, &colWorld);
        MPI_Comm_rank(colWorld, &rankInColWorld);

		// Create fiber world
		std::vector<int> fibRanks = fibGroupRanks[rowRank][colRank];
		MPI_Group_incl(worldGroup, fibRanks.size(), fibRanks.data(), &fibGroup);
		MPI_Comm_create(MPI_COMM_WORLD, fibGroup, &fibWorld);
        MPI_Comm_rank(fibWorld, &rankInFibWorld);
    }
	


	~ProcGrid() {
		//MPI_Group_free(&rowGroup);
		//MPI_Group_free(&colGroup);
		//MPI_Group_free(&fibGroup);
		//MPI_Group_free(&worldGroup);
	}

    void printInfo() {
        std::cout << "Rank: " << myrank 
                  << ",\tRow Rank: " << rowRank 
                  << ",\tCol Rank: " << colRank 
                  << ",\tFib Rank: " << fibRank << std::endl;
    }

    int nProcRow;
    int nProcCol;
    int nProcFib;
    int myrank;
    int nprocs;
    MPI_Group rowGroup;
    MPI_Group colGroup;
    MPI_Group fibGroup;
    MPI_Group worldGroup;
    MPI_Comm rowWorld;
    MPI_Comm colWorld;
    MPI_Comm fibWorld;
    int rankInRowWorld;
    int rankInColWorld;
    int rankInFibWorld;
    int rowRank;
    int colRank;
    int fibRank;
};

class ParMat {
public:
    ParMat(int m, int n, ProcGrid& grid, char frontFace)
        : nRowGlobal(m), nColGlobal(n), grid(grid), frontFace(frontFace) {
        //initialize();
		if (frontFace == 'A') {
            // Distribute matrix rows
            nRowLocal = nRowGlobal / grid.nProcRow;
            localRowStart = nRowLocal * grid.rankInColWorld;
            if (grid.rankInColWorld == grid.nProcRow - 1) {
                nRowLocal = nRowGlobal - nRowLocal * grid.rankInColWorld;
            }

            // Distribute matrix columns
            localColStart = nColGlobal / grid.nProcCol * grid.rankInRowWorld;
            nColLocal = nColGlobal / grid.nProcCol;
            if (grid.rankInRowWorld == grid.nProcCol - 1) {
                nColLocal = nColGlobal - nColGlobal / grid.nProcCol * grid.rankInRowWorld;
            }

            // Further distribute matrix columns to fibers
            localColStart += nColLocal / grid.nProcFib * grid.rankInFibWorld;
            if (grid.rankInFibWorld < grid.nProcFib - 1) {
                nColLocal /= grid.nProcFib;
            } else {
                nColLocal -= nColLocal / grid.nProcFib * grid.rankInFibWorld;
            }
		}
		else if (frontFace == 'B') {
            // Similar logic for frontFace 'B'
            nRowLocal = nRowGlobal / grid.nProcCol;
            localRowStart = nRowLocal * grid.rankInRowWorld;
            if (grid.rankInRowWorld == grid.nProcCol - 1) {
                nRowLocal = nRowGlobal - nRowLocal * grid.rankInRowWorld;
            }

            localColStart = nColGlobal / grid.nProcFib * grid.rankInFibWorld;
            nColLocal = nColGlobal / grid.nProcFib;
            if (grid.rankInFibWorld == grid.nProcFib - 1) {
                nColLocal = nColGlobal - nColGlobal / grid.nProcFib * grid.rankInFibWorld;
            }

            localColStart += nColLocal / grid.nProcRow * grid.rankInColWorld;
            if (grid.rankInColWorld < grid.nProcRow - 1) {
                nColLocal /= grid.nProcRow;
            } else {
                nColLocal -= nColLocal / grid.nProcRow * grid.rankInColWorld;
            }
        }
		else if (frontFace == 'C') {
			// Distribute matrix rows
			nRowLocal = nRowGlobal / grid.nProcRow;
			localRowStart = nRowLocal * grid.rankInColWorld;
			if (grid.rankInColWorld == grid.nProcRow - 1) {
				nRowLocal = nRowGlobal - nRowLocal * grid.rankInColWorld;
			}

			// Distribute matrix columns
			localColStart = nColGlobal / grid.nProcFib * grid.rankInFibWorld;
			nColLocal = nColGlobal / grid.nProcFib;
			if (grid.rankInFibWorld == grid.nProcFib - 1) {
				nColLocal = nColGlobal - nColGlobal / grid.nProcFib * grid.rankInFibWorld;
			}

			// Further distribute matrix columns to grid fibers
			localColStart += nColLocal / grid.nProcCol * grid.rankInRowWorld;
			if (grid.rankInRowWorld < grid.nProcCol - 1) {
				nColLocal /= grid.nProcCol;
			} else {
				nColLocal -= nColLocal / grid.nProcCol * grid.rankInRowWorld;
			}
		}
        //std::cout << nRowLocal << "x" << nColLocal << ", " << sizeof (double) << std::endl;
        //localMat.resize(nRowLocal*nColLocal);
        //localMat = (double *) malloc(nRowLocal * nColLocal * sizeof(double));
        localMat = new double[nRowLocal * nColLocal];
    }

    ~ ParMat(){
        delete[] localMat;
        //free(localMat);
    }

    void generate() {
        //localMat.resize(nRowLocal, std::vector<double>(nColLocal, 0.0));
		for (int idxColLocal = 0; idxColLocal < nColLocal; ++idxColLocal) {
			for (int idxRowLocal = 0; idxRowLocal < nRowLocal; ++idxRowLocal) {
                int idxRowGlobal = localRowStart + idxRowLocal;
                int idxColGlobal = localColStart + idxColLocal;
				int idx = idxColLocal * nRowLocal + idxRowLocal;
                localMat[idx] = (double)(idxRowGlobal * nColGlobal + idxColGlobal);
            }
        }
    }

    void printLocalMatrix() const {
		std::cout << "Local Matrix (Process " << grid.myrank << " [" << nRowLocal << "x" << nColLocal << "]" <<  "):\n";
		//for (const auto& row : localMat) {
			//for (const auto& val : row) {
				//std::cout << val << " ";
			//}
			//std::cout << "\n";
		//}
    }

	int nRowGlobal;
    int nColGlobal;
    int nRowLocal;
    int nColLocal;
    ProcGrid& grid;
    char frontFace;
    int localRowStart;
    int localColStart;
    //std::vector<std::vector<double>> localMat;
    //std::vector<double> localMat;
    double *localMat;
};

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Default values
    int p1 = 0, p2 = 0, p3 = 0;
    int n1 = 0, n2 = 0, n3 = 0;

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
		std::cerr << "Error: Number of processes does not match p1 * p2 * p3." << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

    // Print the parameters for verification (optional)
    if (myrank == 0) {
        std::cout << "Parameters:" << std::endl;
        std::cout << "p1: " << p1 << std::endl;
        std::cout << "p2: " << p2 << std::endl;
        std::cout << "p3: " << p3 << std::endl;
        std::cout << "n1: " << n1 << std::endl;
        std::cout << "n2: " << n2 << std::endl;
        std::cout << "n3: " << n3 << std::endl;
    }

    // Create the process grid
    ProcGrid grid(p1, p2, p3);
    //grid.printInfo();
	
	ParMat A(n1, n2, grid, 'A');
    A.generate();
    A.printLocalMatrix();
    
    //MPI_Barrier(MPI_COMM_WORLD);
    //if(myrank == 0) std::cout << "---" << std::endl;

    //ParMat B(n2, n3, grid, 'B');
	//B.generate();
    //B.printLocalMatrix();

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
