cmake_minimum_required(VERSION 3.16)
project(franka_control_py LANGUAGES CXX)

# --------------------------------------------------
# Global config
# --------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --------------------------------------------------
# Dependencies
# --------------------------------------------------
find_package(Python 3 REQUIRED COMPONENTS Development)
find_package(Franka REQUIRED)
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v1.9.2
)
FetchContent_MakeAvailable(nanobind)

# --------------------------------------------------
# RT core (pure C++, no Python, no linker magic)
# --------------------------------------------------
add_library(rt_core STATIC
  src/controller.cpp
)
target_include_directories(rt_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(rt_core
  PUBLIC
    Franka::Franka
)
# Explicitly no RPATH / linker shenanigans here
set_target_properties(rt_core PROPERTIES
  SKIP_BUILD_RPATH TRUE
)

# -----------------------
# Python extension module
# -----------------------
nanobind_add_module(control
  src/nb_module.cpp
)
target_link_libraries(control
  PRIVATE
    rt_core
)
target_include_directories(control
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
install(TARGETS control
  LIBRARY DESTINATION fr3_mpc
)

# --------------------------------------------------
# RPATH abuse (documented, scoped, owned)
# --------------------------------------------------
# Local user install paths we want the loader to see
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
set(_LOCAL_RPATH "${PROJECT_ROOT}/.local/lib")
set_target_properties(control PROPERTIES
  SKIP_BUILD_RPATH FALSE
  BUILD_RPATH "${_LOCAL_RPATH}"
  INSTALL_RPATH "${_LOCAL_RPATH}"

  # Keep symbols tight — avoid accidental ABI surface
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

# Force DT_RPATH instead of RUNPATH
# We do this *only* for the Python module, on purpose.
target_link_options(control PRIVATE
  -Wl,--disable-new-dtags
)

# --------------------------------------------------
# Notes for future archaeologists:
#
# - RPATH is intentional and relied upon
# - We prefer explicit ELF loader behavior over env vars
# - rt_core must remain Python-free and RT-safe
# - If this breaks, inspect with:
#     readelf -d franka_control*.so | grep RPATH
#
# --------------------------------------------------