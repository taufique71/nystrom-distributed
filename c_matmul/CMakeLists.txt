# Find the MPI package
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(matmul matmul.cpp)
add_executable(nystrom nystrom.cpp)
add_executable(prng_test prng_test.cpp)

# Link the MPI libraries
target_link_libraries(matmul MPI::MPI_CXX)
target_link_libraries(nystrom MPI::MPI_CXX)

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(matmul OpenMP::OpenMP_CXX)
    target_link_libraries(nystrom OpenMP::OpenMP_CXX)
    target_link_libraries(prng_test OpenMP::OpenMP_CXX)
endif()

# Link MKL
if(MKL_FOUND)
    target_link_libraries(matmul ${MKL_LIBRARIES})
    target_link_libraries(nystrom ${MKL_LIBRARIES})
endif()
