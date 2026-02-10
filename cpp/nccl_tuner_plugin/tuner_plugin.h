#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

const char* cclagent_tuner_plugin_version();

// request_json and response_buffer are UTF-8 JSON payloads.
// Returns 0 on success, non-zero on error.
int cclagent_tuner_get_decision(
    const char* request_json,
    char* response_buffer,
    size_t response_buffer_len);

#ifdef __cplusplus
}
#endif
