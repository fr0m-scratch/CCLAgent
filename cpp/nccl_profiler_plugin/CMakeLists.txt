add_library(cclagent_profiler_plugin SHARED
  profiler_plugin.cc
)

target_include_directories(cclagent_profiler_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(cclagent_profiler_plugin PRIVATE cclagent_ipc_common)

set_target_properties(cclagent_profiler_plugin PROPERTIES
  OUTPUT_NAME "cclagent_profiler_plugin"
)
