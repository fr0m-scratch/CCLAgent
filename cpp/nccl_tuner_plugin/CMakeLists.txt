add_library(cclagent_tuner_plugin SHARED
  tuner_plugin.cc
)

target_include_directories(cclagent_tuner_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(cclagent_tuner_plugin PRIVATE cclagent_ipc_common)

set_target_properties(cclagent_tuner_plugin PROPERTIES
  OUTPUT_NAME "cclagent_tuner_plugin"
)
