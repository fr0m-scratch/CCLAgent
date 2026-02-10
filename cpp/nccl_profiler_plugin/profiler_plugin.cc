#include "nccl_profiler_plugin/profiler_plugin.h"

#include <cstring>
#include <mutex>
#include <string>

namespace {

std::mutex g_mu;
std::string g_last_event = "{}";

int WriteResponse(const std::string& response, char* buffer, size_t buffer_len) {
  if (buffer == nullptr || buffer_len == 0) {
    return 2;
  }
  if (response.size() + 1 > buffer_len) {
    return 3;
  }
  std::memcpy(buffer, response.c_str(), response.size() + 1);
  return 0;
}

}  // namespace

extern "C" const char* cclagent_profiler_plugin_version() {
  return "0.1.0";
}

extern "C" int cclagent_profiler_emit(const char* event_json) {
  if (event_json == nullptr) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  g_last_event = event_json;
  return 0;
}

extern "C" int cclagent_profiler_pull(char* response_buffer, size_t response_buffer_len) {
  std::lock_guard<std::mutex> lock(g_mu);
  return WriteResponse(g_last_event, response_buffer, response_buffer_len);
}
