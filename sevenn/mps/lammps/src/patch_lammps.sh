#!/bin/bash
# Patch a LAMMPS source tree with pair style e3gnn for the Apple GPU. usage: patch_lammps.sh LAMMPS_ROOT PYTHON SEVENN_ROOT
# No libtorch link: the model runs in the Python environment PYTHON. Called by `sevenn patch_lammps` on a Mac.
set -e
root="$1"; py="$2"; sevenn_root="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ ! -d "$root/cmake" ] || [ ! -d "$root/src" ]; then echo "Error: $root is not a root of LAMMPS source"; exit 1; fi
if [ -f "$root/src/pair_e3gnn_mps.cpp" ] || [ -f "$root/src/pair_e3gnn.cpp" ]; then
  echo "Error: this LAMMPS is already patched. Use a fresh source tree."; exit 1
fi
mkdir -p "$root/_backups"
cp "$root/cmake/CMakeLists.txt" "$root/_backups/CMakeLists.txt"
cp "$HERE"/pair_e3gnn_mps.cpp "$HERE"/pair_e3gnn_mps.h "$root/src/"
cat >> "$root/cmake/CMakeLists.txt" <<CMAKE

# SevenNet pair style e3gnn on the Apple GPU
set(Python3_EXECUTABLE "$py")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
target_link_libraries(lammps PUBLIC Python3::Python)
target_compile_definitions(lammps PRIVATE SEVENN_DEFAULT_PYTHON="$py" SEVENN_DEFAULT_ROOT="$sevenn_root")
CMAKE
echo "Patched $root: pair_e3gnn_mps.* in src/, Python link in cmake/CMakeLists.txt, original in _backups/"
echo "Example build commands, under LAMMPS root"
echo "  mkdir build; cd build"
echo "  cmake ../cmake -DBUILD_MPI=no -DBUILD_OMP=no -DPKG_MANYBODY=no"
echo "  make -j 4"
