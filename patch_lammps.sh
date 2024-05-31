#!/bin/bash

lammps_root=$1
echo "Usage: sh patch_lammps.sh {lammps_root}"

# Check if the lammps_root directory exists
if [ ! -d "$lammps_root" ]; then
    echo "No such directory: $lammps_root"
    exit 1
fi

# Check if the given directory is the root of LAMMPS source
if [ ! -d "$lammps_root/cmake" ]; then
    echo "Given $lammps_root is not root of LAMMPS source"
    exit 1
fi

# Check if the script is being run from the root of SevenNet
if [ ! -d "./pair_e3gnn" ]; then
    echo "Please run this script in the root of SevenNet"
    exit 1
fi

if [ -f "$lammps_root/src/pair_e3gnn.cpp" ]; then
    echo "Seems like given LAMMPS is already patched."
    exit 0
fi

# Extract LAMMPS version and update
#
lammps_version=$(grep "#define LAMMPS_VERSION" $lammps_root/src/version.h | awk '{print $3, $4, $5}' | tr -d '"')

# Combine version and update
detected_version="$lammps_version"
required_version="2 Aug 2023"  # Example required version

if [[ "$detected_version" != "$required_version" ]]; then
    echo "Warning: Detected LAMMPS version ($detected_version) may not be compatible. Required version: $required_version"
fi

# Create a backup directory if it doesn't exist
backup_dir="$lammps_root/_backups"
mkdir -p $backup_dir

# 1. Copy comm_* from original LAMMPS source as backup
cp $lammps_root/src/comm_brick.cpp $backup_dir/
cp $lammps_root/src/comm_brick.h $backup_dir/

# 2. Copy everything inside pair_e3gnn to LAMMPS source
cp ./pair_e3gnn/* $lammps_root/src/

# 3. Copy cmake/CMakeLists.txt from original source as backup
cp $lammps_root/cmake/CMakeLists.txt $backup_dir/CMakeLists.txt

# 4. Patch cmake/CMakeLists.txt
sed -i "s/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD 14)/" $lammps_root/cmake/CMakeLists.txt
cat >> $lammps_root/cmake/CMakeLists.txt << "EOF2"

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
EOF2


# Check if the command is found and its value is true
cuda_support=$(ompi_info --parsable --all | grep mpi_built_with_cuda_support:value)
if [[ -z "$cuda_support" ]]; then
    echo "OpenMPI not found, parallel performance could be low"
elif [[ "$cuda_support" == *"true" ]]; then
    echo "OpenMPI is CUDA aware"
else
    echo "This OpenMPI is not CUDA aware, parallel performance could be low"
fi

# ?. Print changes and backup file locations
echo "Changes made:"
echo "  - Original LAMMPS files (src/comm_brick.*, cmake/CMakeList.txt) are in {lammps_root}/_backups"
echo "  - Copied contents of pair_e3gnn to $lammps_root/src/"
echo "  - Patched CMakeLists.txt: include LibTorch, CXX_STANDARD 14"

# ?. Provide example cmake command to the user
echo "Example build commends, under LAMMPS root"
echo "  mkdir build; cd build"
echo "  cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`"
echo "  make -j 4"

exit 0

