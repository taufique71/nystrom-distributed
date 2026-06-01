cmake_minimum_required(VERSION 3.14)
project(slate-matmul CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(slate REQUIRED)

add_executable(slate-matmul slate-matmul.cpp)
target_link_libraries(slate-matmul PRIVATE
    slate
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)
