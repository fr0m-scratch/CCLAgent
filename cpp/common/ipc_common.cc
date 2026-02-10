#include "common/ipc_common.h"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace cclagent {
namespace ipc {

std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) {
    return "";
  }
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool WriteFileAtomic(const std::string& path, const std::string& content) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp);
    if (!out.good()) {
      return false;
    }
    out << content;
  }
  return std::rename(tmp.c_str(), path.c_str()) == 0;
}

}  // namespace ipc
}  // namespace cclagent
