cmake_minimum_required(VERSION 3.26)
project(macseek LANGUAGES C CXX OBJCXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_program(CCACHE ccache)
if(CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
  set(CMAKE_OBJCXX_COMPILER_LAUNCHER ${CCACHE})
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=native -DNDEBUG")
set(CMAKE_OBJCXX_FLAGS_RELEASE "-O3 -mcpu=native -DNDEBUG")
set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")

# ---- Metal kernels: no offline Metal toolchain needed: all src/kernels/*.metal are bundled (local
# #include "x.h" inlined once) into build/macseek.metal, compile-checked at build time by `metalc` (newLibraryWithSource,
# Metal 4), and compiled at runtime by Device (cached per process; MTLBinaryArchive later). ----
add_executable(metalc ${CMAKE_SOURCE_DIR}/tools/metalc/metalc.mm)
target_compile_options(metalc PRIVATE -fobjc-arc)
target_link_libraries(metalc PRIVATE "-framework Metal" "-framework Foundation")
file(GLOB METAL_SRCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/*.metal)
file(GLOB METAL_HDRS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/*.h)
set(METALLIB ${CMAKE_BINARY_DIR}/macseek.metal)
add_custom_command(OUTPUT ${METALLIB}
  COMMAND metalc ${METALLIB} ${METAL_SRCS}
  DEPENDS metalc ${METAL_SRCS} ${METAL_HDRS} VERBATIM)
add_custom_target(metallib ALL DEPENDS ${METALLIB})

# ---- engine library: every module dir under src/ except app ----
file(GLOB_RECURSE LIB_SRCS CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/src/core/*.cpp ${CMAKE_SOURCE_DIR}/src/core/*.mm
  ${CMAKE_SOURCE_DIR}/src/gguf/*.cpp ${CMAKE_SOURCE_DIR}/src/store/*.cpp ${CMAKE_SOURCE_DIR}/src/store/*.mm
  ${CMAKE_SOURCE_DIR}/src/kernels/*.cpp ${CMAKE_SOURCE_DIR}/src/kernels/*.mm
  ${CMAKE_SOURCE_DIR}/src/model/*.cpp ${CMAKE_SOURCE_DIR}/src/model/*.mm
  ${CMAKE_SOURCE_DIR}/src/tok/*.cpp ${CMAKE_SOURCE_DIR}/src/sample/*.cpp ${CMAKE_SOURCE_DIR}/src/sample/*.mm
  ${CMAKE_SOURCE_DIR}/src/server/*.cpp)
add_library(macseek_lib STATIC ${LIB_SRCS})
target_include_directories(macseek_lib PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(macseek_lib PUBLIC MACSEEK_METALLIB="${METALLIB}")
target_link_libraries(macseek_lib PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")
add_dependencies(macseek_lib metallib)

file(GLOB APP_SRCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/app/*.cpp ${CMAKE_SOURCE_DIR}/src/app/*.mm)
if(APP_SRCS)
  add_executable(macseek ${APP_SRCS})
  target_link_libraries(macseek PRIVATE macseek_lib)
endif()

# ---- tests: every tests/test_*.cpp|mm is its own executable + ctest; tests/ref/*.cpp is shared CPU reference code ----
enable_testing()
file(GLOB REF_SRCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/ref/*.cpp)
if(REF_SRCS)
  add_library(macseek_ref STATIC ${REF_SRCS})
  target_include_directories(macseek_ref PUBLIC ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/src)
endif()
file(GLOB TEST_SRCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp ${CMAKE_SOURCE_DIR}/tests/test_*.mm)
foreach(t ${TEST_SRCS})
  get_filename_component(n ${t} NAME_WE)
  add_executable(${n} ${t})
  target_link_libraries(${n} PRIVATE macseek_lib)
  if(REF_SRCS)
    target_link_libraries(${n} PRIVATE macseek_ref)
  endif()
  add_test(NAME ${n} COMMAND ${n})
endforeach()

# ---- microbenchmarks: bench/*.cpp|mm ----
file(GLOB BENCH_SRCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/bench/*.cpp ${CMAKE_SOURCE_DIR}/bench/*.mm)
foreach(b ${BENCH_SRCS})
  get_filename_component(n ${b} NAME_WE)
  add_executable(bench_${n} ${b})
  target_link_libraries(bench_${n} PRIVATE macseek_lib)
endforeach()
