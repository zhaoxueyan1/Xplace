add_subdirectory(common)
add_subdirectory(draw_placement)
add_subdirectory(flute_cpp)
set(CUGR2_BUILD_LEGACY_ROUTE OFF CACHE BOOL "" FORCE)
set(CUGR2_EXTERNAL_FLUTE ON CACHE BOOL "" FORCE)
add_subdirectory(${PATH_THIRDPARTY_ROOT}/cu-gr-2/src ${CMAKE_CURRENT_BINARY_DIR}/cu-gr-2-route-core EXCLUDE_FROM_ALL)
add_subdirectory(gpugr)
option(
  XPLACE_BUILD_LEGACY_CUGR
  "Build the legacy Rsyn-backed CUGR extension in addition to CUGR2"
  OFF
)
if(XPLACE_BUILD_LEGACY_CUGR)
  add_subdirectory(cugr)
endif()
add_subdirectory(io_parser)
add_subdirectory(routedp)
add_subdirectory(gputimer)

if (TORCH_ENABLE_CUDA)
  add_subdirectory(dct_cuda)
  add_subdirectory(density_map_cuda)
  add_subdirectory(gpudp)
  add_subdirectory(hpwl_cuda)
  add_subdirectory(wa_wirelength_hpwl_cuda)
  add_subdirectory(wirelength_timing_cuda)
endif()
