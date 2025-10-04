#include "matrix.h"
#include "procgrid.h"
#include "utils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm world = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(world, &rank);

    ProcGrid grid1(2, 2, 1);
    int n = 4;
    std::vector<int> rowDistrib = {2, 2}; // each process row gets 2 rows
    std::vector<int> colDistrib = {2, 2}; // each process col gets 2 columns
    std::string filePath = "/deac/csc/ballardGrp/rahmm224/testdata.bin";

    ParMat A(n, n, grid1, 'A', rowDistrib, colDistrib);
    A.parallelReadBinary(filePath, world);

    for (size_t i = 0; i < 4; i++) {
    std::cout << A.localMat[i] << "\n";
}

    std::cout << std::endl;


    MPI_Finalize();
    return 0;
}