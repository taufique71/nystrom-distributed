# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# User option to override CUDA usage
option(USE_CUDA "Enable CUDA/cuBLAS support" OFF)

# Detect NERSC programming environment
if (DEFINED ENV{PE_ENV})
    set(PE_ENVIRONMENT $ENV{PE_ENV})
    message(STATUS "Detected NERSC Programming Environment: ${PE_ENVIRONMENT}")
	
	if(${PE_ENVIRONMENT} STREQUAL "NVIDIA")
        set(USE_CUDA ON)
    elseif(${PE_ENVIRONMENT} STREQUAL "INTEL")
        set(USE_CUDA OFF)
    endif()

endif()

# Find MPI (required)
find_package(MPI REQUIRED)
if (MPI_FOUND)
	message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
	include_directories(${MPI_INCLUDE_PATH})
else()
	message(FATAL_ERROR "MPI not found")
endif()


# Find OpenMP (required)
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
	message(STATUS "OpenMP found: ${MPI_CXX_COMPILER}")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
	message(FATAL_ERROR "OpenMP not found")
endif()

# Configure BLAS backend (required)
if (USE_CUDA)
	find_package(CUDA REQUIRED)
	if (CUDA_FOUND)
		add_definitions(-DUSE_CUBLAS)
		include_directories(${CUDA_INCLUDE_DIRS})
		link_libraries(${CUDA_LIBRARIES} cublas curand)
		enable_language(CUDA)
		message(STATUS "Using cuBLAS (CUDA)")
	else()
		message(FATAL_ERROR "CUDA not found")
	endif()

else()
	find_package(MKL REQUIRED)
	if (MKL_FOUND)
		add_definitions(-DUSE_MKL)
		include_directories(${MKL_INCLUDE_DIRS})
		link_libraries(${MKL_LIBRARIES})
		message(STATUS "Using MKL (CPU)")
	else()
		message(FATAL_ERROR "MKL not found")
	endif()

endif()



# Add the executable
add_executable(matmul matmul.cpp)
add_executable(nystrom nystrom.cpp)
add_executable(test-prng test-prng.cpp)
add_executable(test-cublas test-cublas.cpp)
add_executable(test-cua-mpi test-cuda-aware-mpi.cpp)

if (USE_CUDA)
    target_link_libraries(matmul OpenMP::OpenMP_CXX MPI::MPI_CXX ${CUDA_LIBRARIES} cublas curand)
    target_link_libraries(nystrom OpenMP::OpenMP_CXX MPI::MPI_CXX ${CUDA_LIBRARIES} cublas)
    target_link_libraries(test-prng OpenMP::OpenMP_CXX ${CUDA_LIBRARIES} curand)
    target_link_libraries(test-cublas OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})
    target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-target-accel=nvidia80>)
    target_compile_options(test-cua-mpi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-target-accel=nvidia80>)
else()
    target_link_libraries(matmul OpenMP::OpenMP_CXX MPI::MPI_CXX ${MKL_LIBRARIES})
    target_link_libraries(nystrom OpenMP::OpenMP_CXX MPI::MPI_CXX ${MKL_LIBRARIES})
    target_link_libraries(test-prng OpenMP::OpenMP_CXX ${MKL_LIBRARIES})
    target_link_libraries(test-cublas OpenMP::OpenMP_CXX ${MKL_LIBRARIES})
endif()

# Debug message to check the value of USE_CUDA
message(STATUS "USE_CUDA is set to: ${USE_CUDA}")
