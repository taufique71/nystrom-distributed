# Find the MPI package
find_package(MPI REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(matmul main.cpp)

# Link the MPI libraries
target_link_libraries(matmul MPI::MPI_CXX)
