set(CUGR_ROOT "${PROJECT_SOURCE_DIR}/thirdparty/cu-gr")

execute_process(
  COMMAND git -C "${CUGR_ROOT}" rev-parse HEAD
  OUTPUT_VARIABLE CUGR_SOURCE_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
string(LENGTH "${CUGR_SOURCE_COMMIT}" CUGR_SOURCE_COMMIT_LENGTH)
if(NOT CUGR_SOURCE_COMMIT_LENGTH EQUAL 40 OR NOT CUGR_SOURCE_COMMIT MATCHES "^[0-9a-fA-F]+$")
  set(CUGR_SOURCE_COMMIT "unknown")
endif()
execute_process(
  COMMAND git -C "${CUGR_ROOT}" status --porcelain --untracked-files=all
  RESULT_VARIABLE CUGR_SOURCE_STATUS_RESULT
  OUTPUT_VARIABLE CUGR_SOURCE_STATUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(CUGR_SOURCE_STATUS_RESULT EQUAL 0 AND CUGR_SOURCE_STATUS STREQUAL "")
  set(CUGR_SOURCE_DIRTY 0)
else()
  set(CUGR_SOURCE_DIRTY 1)
endif()

file(GLOB_RECURSE CUGR_RSYN_SOURCES
  "${CUGR_ROOT}/rsyn/src/*.cpp"
  "${CUGR_ROOT}/rsyn/src/*.cc"
  "${CUGR_ROOT}/rsyn/src/*.c"
)
file(GLOB_RECURSE CUGR_SOURCES
  "${CUGR_ROOT}/src/*.cpp"
  "${CUGR_ROOT}/src/*.cc"
  "${CUGR_ROOT}/src/*.c"
)
list(FILTER CUGR_SOURCES EXCLUDE REGEX "/main\\.cpp$")
list(APPEND CUGR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cugr_api.cpp")

find_library(
  CUGR_LIBLEF
  NAMES lef
  PATHS "${PATH_THIRDPARTY_ROOT}/lefdef/lef58/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(
  CUGR_LIBDEF
  NAMES def
  PATHS "${PATH_THIRDPARTY_ROOT}/lefdef/def58/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
find_package(Boost REQUIRED COMPONENTS system filesystem program_options)

# CUGR and Xplace both use short global namespaces (db, gr, and utils).  The
# router is linked into the same Python process, so compile the CUGR sources
# under private names instead of allowing ODR collisions with Xplace GGR.
add_library(cugr_core STATIC ${CUGR_RSYN_SOURCES} ${CUGR_SOURCES})
set_target_properties(cugr_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(cugr_core PRIVATE
  db=cugr_db
  gr=cugr_gr
  utils=cugr_utils
  CUGR_LUT_DIR="${CUGR_ROOT}/src/flute"
  CUGR_SOURCE_COMMIT="${CUGR_SOURCE_COMMIT}"
  CUGR_SOURCE_DIRTY=${CUGR_SOURCE_DIRTY}
)
target_include_directories(cugr_core
  PUBLIC
    "${CUGR_ROOT}/src"
    "${CUGR_ROOT}/rsyn/src"
    "${CUGR_ROOT}/rsyn/src/rsyn/export"
    "${CUGR_ROOT}/rsyn/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  PRIVATE
    "${PATH_THIRDPARTY_ROOT}/lefdef"
)
target_link_libraries(cugr_core
  PUBLIC
    ${CUGR_LIBLEF}
    ${CUGR_LIBDEF}
    Boost::system
    Boost::filesystem
    Boost::program_options
    pthread
)
target_compile_options(cugr_core PRIVATE -fPIC -Werror=return-type)

pybind11_add_module(cugr MODULE cugr_module.cpp)
target_include_directories(cugr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(cugr PRIVATE cugr_core)
target_compile_definitions(cugr PRIVATE TORCH_EXTENSION_NAME=cugr)
target_compile_definitions(cugr PRIVATE
  CUGR_SOURCE_COMMIT="${CUGR_SOURCE_COMMIT}"
  CUGR_SOURCE_DIRTY=${CUGR_SOURCE_DIRTY}
)

install(TARGETS cugr DESTINATION ${XPLACE_LIB_DIR})

# Keep the source license and the LUT provenance with installed CPU runtime
# artifacts. The common Xplace flute LUT copy is installed by the parent
# package, but CUGR's attribution remains part of this target's payload.
install(
  FILES
    "${CUGR_ROOT}/LICENSE"
    "${CUGR_ROOT}/src/flute/license.txt"
  DESTINATION "${XPLACE_LICENSE_DIR}"
)
