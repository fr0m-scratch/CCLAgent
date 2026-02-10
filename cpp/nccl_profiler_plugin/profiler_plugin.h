#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

const char* cclagent_profiler_plugin_version();

// Accepts a JSON event and returns 0 on success.
int cclagent_profiler_emit(const char* event_json);

// Optional pull endpoint for tests/integration.
int cclagent_profiler_pull(char* response_buffer, size_t response_buffer_len);

#ifdef __cplusplus
}
#endif
