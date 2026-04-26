cmake_minimum_required(VERSION 3.18)
project(main LANGUAGES CXX CUDA)

add_executable(main main.cu)

# Pick ONE of these two options:

# Option A — simple (works on most systems)
target_link_libraries(main cublas)

# Option B — explicit toolkit path (more robust)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(main CUDA::cublas)
