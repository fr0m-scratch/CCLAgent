cmake_minimum_required(VERSION 3.16)
project(cclagent_plugins LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(cclagent_ipc_common STATIC
  common/ipc_common.cc
)
target_include_directories(cclagent_ipc_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(nccl_tuner_plugin)
add_subdirectory(nccl_profiler_plugin)
