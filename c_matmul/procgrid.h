#ifndef PROCGRID_H
#define PROCGRID_H

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <omp.h>
#include <mkl.h>

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

#endif
