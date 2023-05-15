#!/bin/sh

export CUDA_VISIBLE_DEVICES=
mpirun -np 2 /TGM/Apps/LAMMPS/lammps/build_testing/lmp < ./in.lmp


