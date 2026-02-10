#include "nccl_tuner_plugin/tuner_plugin.h"

#include <cstring>
#include <string>

namespace {

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

extern "C" const char* cclagent_tuner_plugin_version() {
  return "0.1.0";
}

extern "C" int cclagent_tuner_get_decision(
    const char* request_json,
    char* response_buffer,
    size_t response_buffer_len) {
  (void)request_json;
  const std::string response =
      "{\"status\":\"ok\",\"source\":\"fallback\",\"override\":{},\"reasons\":[\"stub_plugin\"]}";
  return WriteResponse(response, response_buffer, response_buffer_len);
}
