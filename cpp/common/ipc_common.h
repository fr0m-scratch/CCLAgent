#pragma once

#include <string>

namespace cclagent {
namespace ipc {

std::string ReadFile(const std::string& path);
bool WriteFileAtomic(const std::string& path, const std::string& content);

}  // namespace ipc
}  // namespace cclagent
