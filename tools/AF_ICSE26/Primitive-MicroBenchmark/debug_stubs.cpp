// debug_stubs.cpp
//
// Must be compiled as C++ with -std=c++11 (or higher)
// and with an -I pointing at <NCCL_SRC>/src/include.

#include "debug.h"   // this is src/include/debug.h

// 1) The real header says:
//    extern thread_local int ncclDebugNoWarn;
// so we must *define* it as TLS int:
thread_local int ncclDebugNoWarn = 0;

// 2) The real header says:
//    void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...)
// so we define the function (no extern "C", no attribute here):
void ncclDebugLog(ncclDebugLogLevel level,
                  unsigned long    flags,
                  const char*      filefunc,
                  int              line,
                  const char*      fmt, ...)
{
  // no‐op
}