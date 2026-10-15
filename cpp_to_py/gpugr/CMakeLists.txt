# Files
file(GLOB_RECURSE SRC_FILES_GGR ${CMAKE_CURRENT_SOURCE_DIR}/db/*.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/gr/*.cpp)
if (TORCH_ENABLE_CUDA)
    file(GLOB_RECURSE SRC_FILES_GGR_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
else()
    set(SRC_FILES_GGR_CUDA "")
endif()

# CUDA/CPP GGR Kernel
add_library(ggr STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../io_parser/gp/GPDatabase.cpp
                       ${SRC_FILES_GGR}
                       ${SRC_FILES_GGR_CUDA})

set_target_properties(ggr PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (TORCH_ENABLE_CUDA)
    set_target_properties(ggr PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

target_include_directories(ggr PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR})
target_link_libraries(ggr PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common flute pthread)
target_compile_options(ggr PRIVATE -fPIC)
if (TORCH_ENABLE_CUDA)
    target_compile_options(ggr PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")
endif()
target_compile_definitions(ggr PRIVATE
                            ENABLE_CUDA=${TORCH_ENABLE_CUDA})

pybind11_add_module(gpugr MODULE PyBindCppMain.cpp TimingRouteExport.cpp)

target_include_directories(gpugr PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR})
target_link_libraries(gpugr PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common flute io_parser ggr)
target_compile_definitions(gpugr PRIVATE 
                            TORCH_EXTENSION_NAME=gpugr
                            TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
                            TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
                            ENABLE_CUDA=${TORCH_ENABLE_CUDA})


install(TARGETS gpugr DESTINATION ${XPLACE_LIB_DIR})
