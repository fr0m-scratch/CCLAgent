#pragma once

// Placeholder shim for JSON helper inclusion in plugin skeletons.
// Production plugin builds should replace this with a full JSON library
// (for example nlohmann/json.hpp) if structured parsing is required.

namespace cclagent {
namespace json {

inline const char* Version() { return "placeholder"; }

}  // namespace json
}  // namespace cclagent
