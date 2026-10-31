#pragma once
#include <cstdlib>
#include <string>
namespace ms {
inline long env_int(const char* name, long dflt) { const char* v = getenv(name); return v && *v ? strtol(v, nullptr, 10) : dflt; }
inline double env_f(const char* name, double dflt) { const char* v = getenv(name); return v && *v ? strtod(v, nullptr) : dflt; }
inline std::string env_str(const char* name, const char* dflt) { const char* v = getenv(name); return v ? v : dflt; }
}
