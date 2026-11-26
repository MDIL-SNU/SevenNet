/* The LAMMPS headers need the type of MPI_Comm; the plugin makes no MPI call. int is the MPICH type, used by `pip install lammps`. */
#ifndef SEVENN_MPI_STUB_H
#define SEVENN_MPI_STUB_H
typedef int MPI_Comm;
#endif
