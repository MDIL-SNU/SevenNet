# cython: language_level=3
# distutils: language = c++

import pickle
import numpy as np
import lammps.mliap
try:
    import cupy
except ImportError:
    pass
try:
    import torch
except ImportError:
    pass
from libc.stdint cimport uintptr_t

cimport cython
from cpython.ref cimport PyObject
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


cdef extern from "lammps.h" namespace "LAMMPS_NS":
    cdef cppclass LAMMPS:
        pass


cdef extern from "mliap_data_kokkos.h" namespace "LAMMPS_NS":
    cdef cppclass MLIAPDataKokkosDevice:
        # ----- may not need -----
        const double * x
        double H[9]
        int pbcx, pbcy, pbcz
        const long long * tag
        int size_array_rows
        int size_array_cols
        int natoms
        int yoffset
        int zoffset
        int ndims_force
        int ndims_virial
        # -END- may not need -END-
        int size_gradforce
        # ----- write only -----
        double * f
        double * gradforce
        double * betas         # betas for all atoms in list
        double * descriptors   # descriptors for all atoms in list
        double * eatoms         # energies for all atoms in list
        double * energy
        # -END- write only -END-
        int ndescriptors        # number of descriptors
        int nparams             # number of model parameters per element
        int nelements           # number of elements

        # data structures for grad-grad list (gamma)

        # ----- ignore for now -----
        int gamma_nnz           # number of non-zero entries in gamma
        double * gamma         # gamma element
        int * gamma_row_index  # row (parameter) index
        int * gamma_col_index  # column (descriptor) index
        double * egradient      # energy gradient w.r.t. parameters
        # -END- ignore for now -END-

        # data structures for mliap neighbor list
        # only neighbors strictly inside descriptor cutoff

        int ntotal              # total number of owned and ghost atoms on this proc
        int nlistatoms          # current number of atoms in local atom lists
        int nlocal
        int natomneigh          # current number of atoms and ghosts in atom neighbor arrays
        int * numneighs         # neighbors count for each atom
        int * iatoms            # index of each atom
        int * pair_i            # index of each i atom for each ij pair
        int * ielems            # element of each atom
        int nneigh_max          # number of ij neighbors allocated
        int npairs              # number of ij neighbor pairs
        int * jatoms            # index of each neighbor
        int * jelems            # element of each neighbor
        int * elems             # element of each atom in or not in the neighborlist
        double * rij           # distance vector of each neighbor
        # ----- write only -----
        double * graddesc     # descriptor gradient w.r.t. each neighbor
        # -END- write only -END-
        int eflag               # indicates if energy is needed
        int vflag               # indicates if virial is needed
        void * pairmliap        # pointer to base class
        int dev

        void forward_exchange[CommType]  (CommType * copy_from, CommType * copy_to, int vec_len) except +
        void reverse_exchange[CommType] (CommType * copy_from, CommType * copy_to, int vec_len) except +

cdef extern from "mliap_unified_kokkos.h" namespace "LAMMPS_NS":
    cdef cppclass MLIAPDummyDescriptor:
        MLIAPDummyDescriptor(PyObject *, LAMMPS *) except +
        int ndescriptors    # number of descriptors
        int nelements       # # of unique elements
        char *elements     # names of unique elements
        double cutmax       # maximum cutoff needed
        double rcutfac
        double *radelem     # element radii

        void compute_descriptors(MLIAPDataKokkosDevice *)
        void compute_forces(MLIAPDataKokkosDevice *)
        void set_elements(char **, int)

    cdef cppclass MLIAPDummyModel:
        MLIAPDummyModel(PyObject *, LAMMPS *, char * = NULL) except +
        int ndescriptors    # number of descriptors
        int nparams         # number of parameters per element
        int nelements;      # # of unique elements

        void compute_gradients(MLIAPDataKokkosDevice *)

    cdef void update_pair_energy(MLIAPDataKokkosDevice *, double *) except +
    cdef void update_pair_forces(MLIAPDataKokkosDevice *, double *) except +
    cdef void update_atom_energy(MLIAPDataKokkosDevice *, double *) except +
    cdef void set_atom_forces(MLIAPDataKokkosDevice*, double*) except +
    cdef void set_total_energy(MLIAPDataKokkosDevice*, double) except +
    cdef void add_global_virial(MLIAPDataKokkosDevice*, const double*) except +
    cdef void add_global_stress(MLIAPDataKokkosDevice *, const double *) except +


LOADED_MODEL = None


# @property sans getter
def write_only_property(fset):
    return property(fget=None, fset=fset)

cdef create_array(device, void *pointer, shape,is_int):
    size=1
    for i in shape:
        size = size*i
    if ( device == 1):
        mem = cupy.cuda.UnownedMemory(ptr=int( <uintptr_t> pointer), owner=None, size=size)
        memptr = cupy.cuda.MemoryPointer(mem, 0)
        type=cupy.double
        if (is_int):
            type=cupy.int32
        return cupy.ndarray(shape, type, memptr=memptr)
    else:
        if (len(shape) == 1 ):
            if (is_int):
                return np.asarray(<int[:shape[0]]>pointer)
            else:
                return np.asarray(<double[:shape[0]]>pointer)
        else:
            if (is_int):
                return np.asarray(<int[:shape[0],:shape[1]]>pointer)
            else:
                return np.asarray(<double[:shape[0],:shape[1]]>pointer)



# Cython implementation of MLIAPData
# Automatically converts between C arrays and numpy when needed
cdef class MLIAPDataPy:
    cdef MLIAPDataKokkosDevice * data

    def __cinit__(self):
        self.data = NULL
    def get_positions(self):
        """Return (N,3) float64 array of atomic positions (owned + ghosts)."""
        if self.data.x == NULL:
            raise RuntimeError("positions pointer (x) is NULL; backend must fill MLIAPDataKokkosDevice::x")
        # ntotal x 3 view
        return create_array(self.data.dev, <void*> self.data.x, [self.ntotal, 3], False)

    # optional alias some wrappers use
    def get_x(self):
        return self.get_positions()

    def get_edge_index(self):
        """Return (2,E) int64 array of pair indices (i,j)."""
        pi = self.pair_i
        ja = self.jatoms
        if pi is None or ja is None:
            raise RuntimeError("pair_i or jatoms is NULL; backend must fill them")
        # If device arrays (CuPy), stack with CuPy; else with NumPy
        try:
            import cupy as cp
            if isinstance(pi, cp.ndarray) or isinstance(ja, cp.ndarray):
                return cp.stack([pi, ja], axis=0)
        except Exception:
            pass
        return np.stack([np.asarray(pi), np.asarray(ja)], axis=0)

    def get_edge_vectors(self):
        """Return (E,3) float64 array of rij vectors."""
        R = self.rij
        if R is None:
            raise RuntimeError("rij is NULL; backend must fill MLIAPDataKokkosDevice::rij")
        return R

    def get_types(self):
        """Return (N,) int64 array of LAMMPS type/element per atom."""
        arr = self.elems
        if arr is None:
            raise RuntimeError("elems is NULL; backend must fill MLIAPDataKokkosDevice::elems")
        return arr

    def get_atomic_numbers(self):
        """Proxy to get_types(); SevenNet wrapper maps types -> Z with type_to_Z."""
        return self.get_types()
    
    def update_atom_energy(self, ei):
        cdef uintptr_t ptr = 0
        cdef bint got_ptr = False
        cdef double[:] view 

        try:
            ptr = ei.data.ptr
            got_ptr = True
        except Exception:
            pass

        if not got_ptr:
            try:
                ptr = ei.data_ptr()
                got_ptr = True
            except Exception:
                pass

        if got_ptr:
            update_atom_energy(self.data, <double*>ptr)
            return

        view = np.ascontiguousarray(np.asarray(ei, dtype=np.float64))
        update_atom_energy(self.data, &view[0])

    def set_atom_forces(self, Fi):
        cdef uintptr_t ptr
        cdef bint got_ptr = False
        cdef double[:, ::1] view2d
        try:
            ptr = Fi.data.ptr           # CuPy
            got_ptr = True
        except Exception:
            pass
        if not got_ptr:
            try:
                ptr = Fi.data_ptr()     # Torch
                got_ptr = True
            except Exception:
                pass
        if got_ptr:
            set_atom_forces(self.data, <double*>ptr)
            return
        view2d = np.ascontiguousarray(np.asarray(Fi, dtype=np.float64))
        set_atom_forces(self.data, &view2d[0][0])

    def set_total_energy(self, val):
        set_total_energy(self.data, <double>val)

    def add_global_virial(self, v6):
        cdef double[::1] view = np.ascontiguousarray(v6, dtype=np.float64)
        add_global_virial(self.data, &view[0])
    
    def add_global_stress(self, s6):
        cdef double[::1] view = np.ascontiguousarray(s6, dtype=np.float64)
        add_global_stress(self.data, &view[0])

    def get_cell(self):
        return np.array([[self.data.H[0], self.data.H[1], self.data.H[2]],
                         [self.data.H[3], self.data.H[4], self.data.H[5]],
                         [self.data.H[6], self.data.H[7], self.data.H[8]]],
                         dtype=np.float64)

    def get_pbc(self):
        return (bool(self.data.pbcx), bool(self.data.pbcy), bool(self.data.pbcz))

    def get_tags(self):
        if self.data.tag == NULL:
            raise RuntimeError("tag pointer is NULL; backend must fill MLIAPDataKokkosDevice::tag")
        return create_array(self.data.dev, <void*> self.data.tag, [self.ntotal], True)

    def update_pair_energy_cpu(self, eij):
        cdef double[:] eij_arr
        try:
            eij_arr = eij
        except:
            eij_arr = eij.detach().numpy().astype(np.double)
        update_pair_energy(self.data, &eij_arr[0])
    def update_pair_energy_gpu(self, eij):
        cdef uintptr_t ptr;
        try:
            ptr = eij.data.ptr
        except:
            ptr = eij.data_ptr()
        update_pair_energy(self.data, <double*>ptr)
    def update_pair_energy(self, eij):
        if self.data.dev==0:
            self.update_pair_energy_cpu(eij)
        else:
            self.update_pair_energy_gpu(eij)

    def update_pair_forces_cpu(self, fij):
        cdef double[:, ::1] fij_arr
        try:
            fij_arr = fij
        except:
            fij_arr = fij.detach().numpy().astype(np.double)
        update_pair_forces(self.data, &fij_arr[0][0])
    def update_pair_forces_gpu(self, fij):
        cdef uintptr_t ptr
        try:
            ptr = fij.data.ptr
        except:
            ptr = fij.data_ptr()
        update_pair_forces(self.data, <double*>ptr)
    def update_pair_forces(self, fij):
        if self.data.dev==0:
            self.update_pair_forces_cpu(fij)
        else:
            self.update_pair_forces_gpu(fij)

    def forward_exchange(self, copy_from, copy_to, vec_len):
        cdef uintptr_t copy_from_ptr, copy_to_ptr;

        copy_from_dtype = copy_from.dtype
        copy_to_dtype = copy_to.dtype
        if copy_from_dtype != copy_to_dtype:
            raise TypeError(f"Types of ({copy_from_dtype})copy_from and ({copy_to_dtype})copy_to mismatch")

        try:
            #Attempt assuming PyTorch data
            copy_from_ptr = copy_from.data_ptr()
            copy_to_ptr = copy_to.data_ptr()

            if copy_from_dtype == torch.float32:
                self.data.forward_exchange( <float*>copy_from_ptr, <float*>copy_to_ptr, vec_len)
            elif copy_from_dtype == torch.float64:
                self.data.forward_exchange( <double*>copy_from_ptr, <double*>copy_to_ptr, vec_len)
            else:
                raise TypeError(f"Unsupported comms type: ({copy_from_dtype})")
        except:
            #Attempt assuming Numpy data
            copy_from_ptr = copy_from.data.ptr
            copy_to_ptr = copy_to.data.ptr

            if copy_from_dtype == np.float32:
                self.data.forward_exchange( <float*>copy_from_ptr, <float*>copy_to_ptr, vec_len)
            elif copy_from_dtype == np.float64:
                self.data.forward_exchange( <double*>copy_from_ptr, <double*>copy_to_ptr, vec_len)
            else:
                raise TypeError(f"Unsupported comms type: ({copy_from_dtype})")

    def reverse_exchange(self, copy_from, copy_to, vec_len):
        cdef uintptr_t copy_from_ptr, copy_to_ptr;

        copy_from_dtype = copy_from.dtype
        copy_to_dtype = copy_to.dtype
        if copy_from_dtype != copy_to_dtype:
            raise TypeError(f"Types of ({copy_from_dtype})copy_from and ({copy_to_dtype})copy_to mismatch")

        try:
            #Attempt assuming PyTorch data
            copy_from_ptr = copy_from.data_ptr()
            copy_to_ptr = copy_to.data_ptr()

            if copy_from_dtype == torch.float32:
                self.data.reverse_exchange( <float*>copy_from_ptr, <float*>copy_to_ptr, vec_len)
            elif copy_from_dtype == torch.float64:
                self.data.reverse_exchange( <double*>copy_from_ptr, <double*>copy_to_ptr, vec_len)
            else:
                raise TypeError(f"Unsupported comms type: ({copy_from_dtype})")
        except:
            #Attempt assuming Numpy data
            copy_from_ptr = copy_from.data.ptr
            copy_to_ptr = copy_to.data.ptr

            if copy_from_dtype == np.float32:
                self.data.reverse_exchange( <float*>copy_from_ptr, <float*>copy_to_ptr, vec_len)
            elif copy_from_dtype == np.float64:
                self.data.reverse_exchange( <double*>copy_from_ptr, <double*>copy_to_ptr, vec_len)
            else:
                raise TypeError(f"Unsupported comms type: ({copy_from_dtype})")
        
    @property
    def f(self):
        if self.data.f is NULL:
            return None
        return create_array(self.data.dev, self.data.f, [self.ntotal, 3],False)


    @property
    def size_gradforce(self):
        return self.data.size_gradforce

    @write_only_property
    def gradforce(self, value):
        if self.data.gradforce is NULL:
            raise ValueError("attempt to set NULL gradforce")
        cdef double[:, :] gradforce_view = <double[:self.ntotal, :self.size_gradforce]> &self.data.gradforce[0]
        cdef double[:, :] value_view = value
        gradforce_view[:] = value_view
        print("This code has not been tested or optimized for the GPU, if you are getting this warning optimize gradforce")

    @write_only_property
    def betas(self, value):
        if self.data.betas is NULL:
            raise ValueError("attempt to set NULL betas")
        cdef double[:, :] betas_view = <double[:self.nlistatoms, :self.ndescriptors]> &self.data.betas[0]
        cdef double[:, :] value_view = value
        betas_view[:] = value_view
        print("This code has not been tested or optimized for the GPU, if you are getting this warning optimize ")

    @write_only_property
    def descriptors(self, value):
        if self.data.descriptors is NULL:
            raise ValueError("attempt to set NULL descriptors")
        cdef double[:, :] descriptors_view = <double[:self.nlistatoms, :self.ndescriptors]> &self.data.descriptors[0]
        cdef double[:, :] value_view = value
        descriptors_view[:] = value_view
        print("This code has not been tested or optimized for the GPU, if you are getting this warning optimize descriptors")

    @property
    def eatoms(self):
        if self.data.eatoms is NULL:
            raise ValueError("attempt to set NULL eatoms")
        return create_array(self.data.dev, self.data.eatoms, [self.nlistatoms],False)


    @write_only_property
    def energy(self, value):
        self.data.energy[0] = <double>value

    @property
    def ndescriptors(self):
        return self.data.ndescriptors

    @property
    def nparams(self):
        return self.data.nparams

    @property
    def nelements(self):
        return self.data.nelements

    # data structures for grad-grad list (gamma)

    @property
    def gamma_nnz(self):
        return self.data.gamma_nnz

    @property
    def gamma(self):
        if self.data.gamma is NULL:
            return None
        return create_array(self.data.dev, self.data.gamma, [self.nlistatoms, self.gamma_nnz],False)

    @property
    def gamma_row_index(self):
        if self.data.gamma_row_index is NULL:
            return None
        return create_array(self.data.dev, self.data.gamma_row_index, [self.nlistatoms, self.gamma_nnz],True)

    @property
    def gamma_col_index(self):
        if self.data.gamma_col_index is NULL:
            return None
        return create_array(self.data.dev, self.data.gamma_col_index, [self.nlistatoms, self.gamma_nnz],True)

    @property
    def egradient(self):
        if self.data.egradient is NULL:
            return None
        return create_array(self.data.dev, self.data.egradient, [self.nelements*self.nparams],False)

    # data structures for mliap neighbor list
    # only neighbors strictly inside descriptor cutoff

    @property
    def ntotal(self):
        return self.data.ntotal

    @property
    def elems(self):
        if self.data.elems is NULL:
            return None
        return create_array(self.data.dev, self.data.elems, [self.ntotal],True)

    @property
    def nlistatoms(self):
        return self.data.nlistatoms

    @property
    def nlocal(self):
        return self.data.nlocal

    @property
    def natomneigh(self):
        return self.data.natomneigh

    @property
    def numneighs(self):
        if self.data.numneighs is NULL:
            return None
        return create_array(self.data.dev, self.data.numneighs, [self.natomneigh],False)

    @property
    def iatoms(self):
        if self.data.iatoms is NULL:
            return None
        return create_array(self.data.dev, self.data.iatoms, [self.natomneigh],True)

    @property
    def ielems(self):
        if self.data.ielems is NULL:
            return None
        return create_array(self.data.dev, self.data.ielems, [self.natomneigh],True)

    @property
    def npairs(self):
        return self.data.npairs

    @property
    def pair_i(self):
        if self.data.pair_i is NULL:
            return None
        return create_array(self.data.dev, self.data.pair_i, [self.npairs],True)

    @property
    def pair_j(self):
        return self.jatoms

    @property
    def jatoms(self):
        if self.data.jatoms is NULL:
            return None
        return create_array(self.data.dev, self.data.jatoms, [self.npairs],True)

    @property
    def jelems(self):
        if self.data.jelems is NULL:
            return None
        return create_array(self.data.dev, self.data.jelems, [self.npairs],True)


    @property
    def rij(self):
        if self.data.rij is NULL:
            return None
        return create_array(self.data.dev, self.data.rij, [self.npairs,3],False)

    @property
    def rij_max(self):
        if self.data.rij is NULL:
            return None
        return create_array(self.data.dev, self.data.rij, [self.nneigh_max,3], False)

    @property
    def nneigh_max(self):
        return self.data.nneigh_max

    @write_only_property
    def graddesc(self, value):
        if self.data.graddesc is NULL:
            raise ValueError("attempt to set NULL graddesc")
        cdef double[:, :, :] graddesc_view = <double[:self.npairs, :self.ndescriptors, :3]> &self.data.graddesc[0]
        cdef double[:, :, :] value_view = value
        graddesc_view[:] = value_view

    @property
    def eflag(self):
        return self.data.eflag

    @property
    def vflag(self):
        return self.data.vflag


# Interface between C and Python compute functions
cdef class MLIAPUnifiedInterfaceKokkos:
    cdef MLIAPDummyModel * model
    cdef MLIAPDummyDescriptor * descriptor
    cdef unified_impl

    def __init__(self, unified_impl):
        self.model = NULL
        self.descriptor = NULL
        self.unified_impl = unified_impl

    def compute_gradients(self, data):
        self.unified_impl.compute_gradients(data)

    def compute_descriptors(self, data):
        self.unified_impl.compute_descriptors(data)

    def compute_forces(self, data):
        self.unified_impl.compute_forces(data)


cdef public void compute_gradients_python_kokkos(unified_int, MLIAPDataKokkosDevice *data) except * with gil:
    pydata = MLIAPDataPy()
    pydata.data = data
    unified_int.compute_gradients(pydata)


cdef public void compute_descriptors_python_kokkos(unified_int, MLIAPDataKokkosDevice *data) except * with gil:
    pydata = MLIAPDataPy()
    pydata.data = data
    unified_int.compute_descriptors(pydata)


cdef public void compute_forces_python_kokkos(unified_int, MLIAPDataKokkosDevice *data) except * with gil:
    pydata = MLIAPDataPy()
    pydata.data = data
    unified_int.compute_forces(pydata)


# Create a MLIAPUnifiedInterface and connect it to the dummy model, descriptor
cdef public object mliap_unified_connect_kokkos(char *fname, MLIAPDummyModel * model,
                                                MLIAPDummyDescriptor * descriptor) with gil:
    str_fname = fname.decode('utf-8')
    if str_fname == 'EXISTS':
        if LOADED_MODEL is None:
            raise ValueError("No unified model loaded")
        unified = LOADED_MODEL
    elif str_fname.endswith(".pt") or str_fname.endswith('.pth'):
        import torch
        unified = torch.load(str_fname,weights_only=False)
    else:
        with open(str_fname, 'rb') as pfile:
            unified = pickle.load(pfile)

    unified_int = MLIAPUnifiedInterfaceKokkos(unified)
    unified_int.model = model
    unified_int.descriptor = descriptor

    unified.interface = unified_int

    if unified.ndescriptors is None:
        raise ValueError("no descriptors set")

    unified_int.descriptor.ndescriptors = <int>unified.ndescriptors
    unified_int.descriptor.rcutfac = <double>unified.rcutfac
    unified_int.model.ndescriptors = <int>unified.ndescriptors
    unified_int.model.nparams = <int>unified.nparams

    if unified.element_types is None:
        raise ValueError("no element type set")

    cdef int nelements = <int>len(unified.element_types)
    cdef char **elements = <char**>malloc(nelements * sizeof(char*))
    cdef char * c_str
    cdef char * s
    cdef ssize_t slen

    if not elements:
        raise MemoryError("failed to allocate memory for element names")

    for i, elem in enumerate(unified.element_types):
        py_str = elem.encode('UTF-8')
        s = py_str
        slen = len(py_str)
        c_str = <char *>malloc((slen+1)*sizeof(char))
        if not c_str:
            raise MemoryError("failed to allocate memory for element names")
        memcpy(c_str, s, slen)
        c_str[slen] = 0
        elements[i] = c_str

    unified_int.descriptor.set_elements(elements, nelements)
    unified_int.model.nelements = nelements

    if hasattr(unified, "cutoff") and unified.cutoff is not None:
        unified_int.descriptor.cutmax = <double> unified.cutoff
    else:
        raise ValueError("Need unified.cutoff to set descriptor.cutmax for scalar-cutoff path")

    for i, elem in enumerate(unified.element_types):
        free(elements[i])
    free(elements)
    return unified_int


# For pre-loading a Python model
def load_from_python(unified):
    global LOADED_MODEL
    LOADED_MODEL = unified
