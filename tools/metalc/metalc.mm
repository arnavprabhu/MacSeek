// metalc <out.metal> <src1.metal> ... : bundle Metal sources (inline local #include "x.h" once each), compile-check the
// bundle with the system Metal compiler (newLibraryWithSource, Metal 4.0), write the bundle only if it compiles.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
static std::set<std::string> seen;
static bool inline_file(const std::string& path, std::string& out) {
  std::ifstream f(path); if (!f) { fprintf(stderr, "metalc: cannot read %s\n", path.c_str()); return false; }
  std::string dir = path.substr(0, path.find_last_of('/') + 1), line;
  out += "// ---- " + path + "\n";
  int ln = 0;
  while (std::getline(f, line)) {
    ln++;
    size_t a = line.find("#include \"");
    if (a != std::string::npos && line.find_first_not_of(" \t") == a) {
      size_t b = line.find('"', a + 10);
      std::string inc = dir + line.substr(a + 10, b - a - 10);
      if (!seen.count(inc)) { seen.insert(inc); if (!inline_file(inc, out)) return false; }
      out += "#line " + std::to_string(ln + 1) + " \"" + path + "\"\n";
      continue;
    }
    if (line.find("#pragma once") != std::string::npos) { out += "\n"; continue; }
    out += line + "\n";
  }
  return true;
}
int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 2) { fprintf(stderr, "usage: metalc out.metal srcs...\n"); return 2; }
    std::string bundle;
    for (int i = 2; i < argc; i++) { std::string p = argv[i]; seen.insert(p); bundle += "#line 1 \"" + p + "\"\n"; if (!inline_file(p, bundle)) return 1; }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_0;
    o.mathMode = MTLMathModeFast;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:bundle.c_str()] options:o error:&err];
    if (!lib) { fprintf(stderr, "metalc: compile FAILED:\n%s\n", err.localizedDescription.UTF8String); return 1; }
    if (err) fprintf(stderr, "metalc: warnings:\n%s\n", err.localizedDescription.UTF8String);
    std::ofstream(argv[1]) << bundle;
    printf("metalc: %lu kernels OK -> %s\n", (unsigned long)lib.functionNames.count, argv[1]);
  }
  return 0;
}
