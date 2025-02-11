cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Neuralnetwork LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)

include_directories(${CMAKE_SOURCE_DIR})
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_DEBUG "-g")

set(SOURCE_FILES
    main.cu
    neuralnetwork.cu
    read_and_process_csv.cpp
)

add_executable(Neuralnetwork ${SOURCE_FILES})

set_target_properties(Neuralnetwork PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(Neuralnetwork ${CUDA_LIBRARIES} ${CUDA_cublasLt_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cudart_LIBRARY} ${CUDA_cublas_LIBRARY})

set_target_properties(Neuralnetwork PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
 