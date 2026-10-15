file(GLOB_RECURSE SRC_FILES_IO_PARSER ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Other extensions link this target, but it does not embed a Python interpreter.
add_library(io_parser SHARED ${SRC_FILES_IO_PARSER})
pybind11_extension(io_parser)
set_target_properties(io_parser PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(
    io_parser PUBLIC ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS})
target_link_libraries(
    io_parser PRIVATE pybind11::module torch ${TORCH_PYTHON_LIBRARY} xplace_common)
target_compile_definitions(io_parser PRIVATE 
    TORCH_EXTENSION_NAME=io_parser
    TORCH_MAJOR_VERSION=${TORCH_MAJOR_VERSION}
    TORCH_MINOR_VERSION=${TORCH_MINOR_VERSION}
    ENABLE_CUDA=0)
target_compile_options(io_parser PRIVATE -fPIC)

install(TARGETS 
    io_parser
    DESTINATION ${XPLACE_LIB_DIR})
