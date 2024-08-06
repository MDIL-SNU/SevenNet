#!/bin/bash

lammps_root=$1
cxx_standard=$2  # 14, 17
d3_support=$3    # 1, 0
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
echo "Usage: sh patch_lammps.sh {lammps_root} {cxx_standard} {d3_support}"

###########################################
# Check if the given arguments are valid  #
###########################################

# Check if the lammps_root directory exists
if [ ! -d "$lammps_root" ]; then
    echo "No such directory: $lammps_root"
    exit 1
fi

# Check if the given directory is the root of LAMMPS source
if [ ! -d "$lammps_root/cmake" ] && [ ! -d "$lammps_root/potentials" ]; then
    echo "Given $lammps_root is not a root of LAMMPS source"
    exit 1
fi

# Check if the script is being run from the root of SevenNet
if [ ! -f "${SCRIPT_DIR}/pair_e3gnn.cpp" ]; then
    echo "Script executed in a wrong directory"
    exit 1
fi

# Check if the patch is already applied
if [ -f "$lammps_root/src/pair_e3gnn.cpp" ]; then
    echo "Seems like given LAMMPS is already patched."
    echo "Example build commends, under LAMMPS root"
    echo "  mkdir build; cd build"
    echo "  cmake ../cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
    echo "  make -j 4"
    exit 0
fi

# Check if the OpenMPI is CUDA aware
cuda_support=$(ompi_info --parsable --all | grep mpi_built_with_cuda_support:value)
if [[ -z "$cuda_support" ]]; then
    echo "OpenMPI not found, parallel performance is not optimal"
elif [[ "$cuda_support" == *"true" ]]; then
    echo "OpenMPI is CUDA aware"
else
    echo "This system's OpenMPI is not 'CUDA aware', parallel performance is not optimal"
fi

# Extract LAMMPS version and update
lammps_version=$(grep "#define LAMMPS_VERSION" $lammps_root/src/version.h | awk '{print $3, $4, $5}' | tr -d '"')

# Combine version and update
detected_version="$lammps_version"
required_version="2 Aug 2023"  # Example required version

# Check if the detected version is compatible
if [[ "$detected_version" != "$required_version" ]]; then
    echo "Warning: Detected LAMMPS version ($detected_version) may not be compatible. Required version: $required_version"
fi

###########################################
# Backup original LAMMPS source code      #
###########################################

# Create a backup directory if it doesn't exist
backup_dir="$lammps_root/_backups"
mkdir -p $backup_dir

# Copy comm_* from original LAMMPS source as backup
cp $lammps_root/src/comm_brick.cpp $backup_dir/
cp $lammps_root/src/comm_brick.h $backup_dir/

# Copy cmake/CMakeLists.txt from original source as backup
cp $lammps_root/cmake/CMakeLists.txt $backup_dir/CMakeLists.txt

###########################################
# Patch LAMMPS source code: e3gnn         #
###########################################

# 1. Copy pair_e3gnn files to LAMMPS source
cp $SCRIPT_DIR/{pair_e3gnn,pair_e3gnn_parallel,comm_brick}.cpp $lammps_root/src/
cp $SCRIPT_DIR/{pair_e3gnn,pair_e3gnn_parallel,comm_brick}.h $lammps_root/src/

# 2. Patch cmake/CMakeLists.txt
sed -i "s/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD $cxx_standard)/" $lammps_root/cmake/CMakeLists.txt
cat >> $lammps_root/cmake/CMakeLists.txt << "EOF2"

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
EOF2

###########################################
# Patch LAMMPS source code: d3            #
###########################################

if [ "$d3_support" -ne 0 ]; then

# 1. Copy pair_d3 files to LAMMPS source
cp $SCRIPT_DIR/pair_d3.cu $lammps_root/src/
cp $SCRIPT_DIR/pair_d3.h $lammps_root/src/

# 2. Patch cmake/CMakeLists.txt
sed -i "s/project(lammps CXX)/project(lammps CXX CUDA)/" $lammps_root/cmake/CMakeLists.txt
sed -i "s/\${LAMMPS_SOURCE_DIR}\/[^.]*.cpp/\${LAMMPS_SOURCE_DIR}\/[^.]*.cpp  \${LAMMPS_SOURCE_DIR}\/[^.]*.cu/" $lammps_root/cmake/CMakeLists.txt
cat >> $lammps_root/cmake/CMakeLists.txt << "EOF2"

find_package(CUDA)
target_link_libraries(lammps PUBLIC ${CUDA_LIBRARIES} cuda)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fmad=false -O3")
set(CMAKE_CUDA_ARCHITECTURES "50;80;86;89;90")
EOF2

fi

###########################################
# Print changes and backup file locations #
###########################################

# Print changes and backup file locations
echo "Changes made:"
echo "  - Original LAMMPS files (src/comm_brick.*, cmake/CMakeList.txt) are in {lammps_root}/_backups"
echo "  - Copied contents of pair_e3gnn to $lammps_root/src/"
echo "  - Patched CMakeLists.txt: include LibTorch, CXX_STANDARD $cxx_standard"
if [ "$d3_support" -ne 0 ]; then
    echo "  - Copied contents of pair_d3 to $lammps_root/src/"
    echo "  - Patched CMakeLists.txt: include CUDA"
fi

# Provide example cmake command to the user
echo "Example build commends, under LAMMPS root"
echo "  mkdir build; cd build"
echo "  cmake ../cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
echo "  make -j 4"

exit 0
