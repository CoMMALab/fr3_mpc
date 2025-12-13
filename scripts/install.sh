#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# Resolve project root (script lives in scripts/)
# --------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$ROOT_DIR/.local"

echo "==> Project root: $ROOT_DIR"
echo "==> Install prefix: $PREFIX"

# --------------------------------------------------
# Sanity checks
# --------------------------------------------------
if [ ! -e "$ROOT_DIR/third_party/pinocchio/.git" ]; then
  echo "Error: pinocchio submodule not initialized"
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

if [ ! -e "$ROOT_DIR/third_party/libfranka/.git" ]; then
  echo "Error: libfranka submodule not initialized"
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

# --------------------------------------------------
# 1. System dependencies
# --------------------------------------------------
echo "==> Installing system dependencies"

sudo apt-get update
sudo apt-get install -y \
  libpoco-dev \
  libeigen3-dev \
  libssl-dev \
  libboost-all-dev \
  liburdfdom-dev \
  libconsole-bridge-dev \
  cmake \
  ninja-build

# --------------------------------------------------
# 2. Build & install Pinocchio
# --------------------------------------------------
echo "==> Building Pinocchio"

PINOCCHIO_DIR="$ROOT_DIR/third_party/pinocchio"
PINOCCHIO_BUILD="$PINOCCHIO_DIR/build"

mkdir -p "$PINOCCHIO_BUILD"
cd "$PINOCCHIO_BUILD"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_PYTHON_INTERFACE=OFF \
  -DBUILD_TESTING=OFF

cmake --build . -j
cmake --install .

# --------------------------------------------------
# 3. Build & install libfranka
# --------------------------------------------------
echo "==> Building libfranka"

LIBFRANKA_DIR="$ROOT_DIR/third_party/libfranka"
LIBFRANKA_BUILD="$LIBFRANKA_DIR/build"

mkdir -p "$LIBFRANKA_BUILD"
cd "$LIBFRANKA_BUILD"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_TESTS=OFF

cmake --build . -j
cmake --install .

# --------------------------------------------------
# 4. Build & install pylibfranka (Python)
# --------------------------------------------------
echo "==> Installing pylibfranka"
(
  cd "$ROOT_DIR/third_party/libfranka"

  # Phase 1: install pybind11 only
  uv pip install pybind11

  # Phase 2: discover pybind11 CMake dir
  PYBIND11_CMAKE_DIR="$(python -m pybind11 --cmakedir)"

  # Phase 3: build pylibfranka with scoped env
  CMAKE_PREFIX_PATH="$PREFIX:$PYBIND11_CMAKE_DIR" \
  LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64" \
  uv pip install .
)
