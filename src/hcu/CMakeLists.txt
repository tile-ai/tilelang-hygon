# HCU backend: target helpers, codegen, transforms, and TVM HIP
# runtime (USE_ROCM=ON — TVM's switch for HIP device runtime).
#
# ROCm ALWAYS sources (e.g. target_utils.cc for TargetIsRocm) stay in
# src/rocm/CMakeLists.txt and are still compiled when USE_HCU=ON; this file
# only owns HCU sources. USE_ROCM pulls in TVM's HIP runtime
# (src/runtime/rocm/*) linked against libamdhip64 from the DTK SDK.
#
# HCU does not use hip stubs: the build host must have DTK installed.
file(GLOB TILE_LANG_HCU_ALWAYS_SRCS
  src/hcu/target_utils.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_HCU_ALWAYS_SRCS})

if(NOT USE_HCU)
  return()
endif()

message(STATUS "HCU backend enabled (DTK SDK required at build time)")

if(NOT USE_ROCM)
  message(FATAL_ERROR
    "USE_HCU=ON requires USE_ROCM=ON (TVM HIP runtime for hsaco load/launch). "
    "Install DTK on the build host.")
endif()

set(CMAKE_HIP_STANDARD 17)
add_compile_definitions(__HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__=1)

include(${TVM_SOURCE}/cmake/utils/FindROCM.cmake)
find_rocm(${USE_ROCM})

if(NOT ROCM_FOUND)
  message(FATAL_ERROR "HCU build requires DTK SDK.")
endif()

message(STATUS "HCU build: using DTK headers from ${ROCM_INCLUDE_DIRS}")
list(APPEND TILE_LANG_INCLUDES ${ROCM_INCLUDE_DIRS})

file(GLOB TILE_LANG_HCU_ACTIVE_SRCS
  src/hcu/codegen/*.cc
  src/hcu/op/*.cc
  src/hcu/transform/*.cc
  src/hcu/utils/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_HCU_ACTIVE_SRCS})
