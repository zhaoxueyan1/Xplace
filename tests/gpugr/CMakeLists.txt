cmake_minimum_required(VERSION 3.18)
project(maze_route_records_test LANGUAGES CXX CUDA)

enable_testing()
add_executable(maze_route_records maze_route_records.cu)
target_compile_features(maze_route_records PRIVATE cuda_std_17)
target_compile_options(maze_route_records PRIVATE -UNDEBUG)
target_include_directories(maze_route_records PRIVATE ../../cpp_to_py/gpugr)
add_test(NAME maze_route_records COMMAND maze_route_records)
