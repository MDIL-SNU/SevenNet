/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Matt Bettencourt (NVIDIA)
------------------------------------------------------------------------- */

#ifdef MLIAP_PYTHON

#include "mliap_unified_kokkos.h"
#include <Python.h>

#include "error.h"
#include "lmppython.h"
#include "memory.h"
#include "mliap_data.h"
#include "mliap_unified_couple_kokkos.h"
#include "pair_mliap.h"
#include "python_compat.h"
#include "utils.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
MLIAPDummyDescriptorKokkos<DeviceType>::MLIAPDummyDescriptorKokkos(LAMMPS *_lmp) :
  Pointers(_lmp), MLIAPDummyDescriptor(_lmp), MLIAPDescriptorKokkos<DeviceType>(lmp, this) {}

template <class DeviceType>
MLIAPDummyDescriptorKokkos<DeviceType>::~MLIAPDummyDescriptorKokkos()
{
  // done in base class
  // Py_DECREF(unified_interface);
}

/* ----------------------------------------------------------------------
   invoke compute_descriptors from Cython interface
   ---------------------------------------------------------------------- */

template <class DeviceType>
void MLIAPDummyDescriptorKokkos<DeviceType>::compute_descriptors(class MLIAPData *data)
{
  PyGILState_STATE gstate = PyGILState_Ensure();
  auto *kokkos_data = dynamic_cast<MLIAPDataKokkos<DeviceType>*>(data);
  MLIAPDataKokkosDevice raw_data(*kokkos_data);
  compute_descriptors_python_kokkos(unified_interface, &raw_data);
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
    PyGILState_Release(gstate);
    lmp->error->all(FLERR, "Running mliappy unified compute_descriptors failure.");
  }
  PyGILState_Release(gstate);
}

/* ----------------------------------------------------------------------
   invoke compute_forces from Cython interface
   ---------------------------------------------------------------------- */

template <class DeviceType>
void MLIAPDummyDescriptorKokkos<DeviceType>::compute_forces(class MLIAPData *data)
{
  PyGILState_STATE gstate = PyGILState_Ensure();
  auto *kokkos_data = dynamic_cast<MLIAPDataKokkos<DeviceType>*>(data);
  MLIAPDataKokkosDevice raw_data(*kokkos_data);
  compute_forces_python_kokkos(unified_interface, &raw_data);
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
    PyGILState_Release(gstate);
    lmp->error->all(FLERR, "Running mliappy unified compute_forces failure.");
  }
  PyGILState_Release(gstate);
}

// not implemented
template <class DeviceType>
void MLIAPDummyDescriptorKokkos<DeviceType>::compute_force_gradients(class MLIAPData *)
{
  error->all(FLERR, "compute_force_gradients not implemented");
}

// not implemented
template <class DeviceType>
void MLIAPDummyDescriptorKokkos<DeviceType>::compute_descriptor_gradients(class MLIAPData *)
{
  error->all(FLERR, "compute_descriptor_gradients not implemented");
}

template <class DeviceType>
void MLIAPDummyDescriptorKokkos<DeviceType>::init()
{
  double cut;
  bool have_scalar_cut = (cutmax > 0.0);
  if (!have_scalar_cut) {
    memory->create(radelem, nelements, "mliap_dummy_descriptor:radelem");
    for (int ielem = 0; ielem < nelements; ielem++) { radelem[ielem] = 2.5; }
    cutmax = 5.0;
  }
  memory->create(cutsq, nelements, nelements, "mliap/descriptor/dummy:cutsq");
  if (have_scalar_cut){
    double cut = cutmax;
    for (int i = 0; i < nelements; ++i) {
      for (int j = 0; j < nelements; ++j) {
        cutsq[i][j] = cut * cut;
      }
    }
  } else{
    for (int ielem = 0; ielem < nelements; ielem++) {
      cut = 5.0 * radelem[ielem] * rcutfac;
      if (cut > cutmax) cutmax = cut;
      cutsq[ielem][ielem] = cut * cut;
      for (int jelem = ielem + 1; jelem < nelements; jelem++) {
        cut = (radelem[ielem] + radelem[jelem]) * rcutfac;
        cutsq[ielem][jelem] = cutsq[jelem][ielem] = cut * cut;
      }
    }
  }
}

template <class DeviceType>
void MLIAPDummyDescriptorKokkos<DeviceType>::set_elements(char **elems, int nelems)
{
  nelements = nelems;
  elements = new char *[nelems];
  for (int i = 0; i < nelems; i++) { elements[i] = utils::strdup(elems[i]); }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
MLIAPDummyModelKokkos<DeviceType>::MLIAPDummyModelKokkos(LAMMPS *lmp, char *coefffilename) :
MLIAPDummyModel(lmp,coefffilename),
MLIAPModelKokkos<DeviceType>(lmp, this)
{
  nonlinearflag = 1;
}

template <class DeviceType>
MLIAPDummyModelKokkos<DeviceType>::~MLIAPDummyModelKokkos()
{
  // manually decrement borrowed reference from Python
  Py_DECREF(unified_interface);
}

template <class DeviceType>
int MLIAPDummyModelKokkos<DeviceType>::get_nparams()
{
  return nparams;
}

template <class DeviceType>
int MLIAPDummyModelKokkos<DeviceType>::get_gamma_nnz(class MLIAPData *)
{
  // TODO: get_gamma_nnz
  return 0;
}

/* ----------------------------------------------------------------------
   invoke compute_gradients from Cython interface
   ---------------------------------------------------------------------- */

template <class DeviceType>
void MLIAPDummyModelKokkos<DeviceType>::compute_gradients(class MLIAPData *data)
{
  PyGILState_STATE gstate = PyGILState_Ensure();
  auto *kokkos_data = dynamic_cast<MLIAPDataKokkos<DeviceType>*>(data);
  MLIAPDataKokkosDevice raw_data(*kokkos_data);
  compute_gradients_python_kokkos(unified_interface, &raw_data);
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
    PyGILState_Release(gstate);
    MLIAPModelKokkos<DeviceType>::error->all(FLERR, "Running mliappy unified compute_gradients failure.");
  }
  PyGILState_Release(gstate);
}

// not implemented
template <class DeviceType>
void MLIAPDummyModelKokkos<DeviceType>::compute_gradgrads(class MLIAPData *)
{
  MLIAPModelKokkos<DeviceType>::error->all(FLERR, "compute_gradgrads not implemented");
}

// not implemented
template <class DeviceType>
void MLIAPDummyModelKokkos<DeviceType>::compute_force_gradients(class MLIAPData *)
{
  MLIAPModelKokkos<DeviceType>::error->all(FLERR, "compute_force_gradients not implemented");
}

/* ----------------------------------------------------------------------
   memory usage unclear due to Cython/Python implementation
   ---------------------------------------------------------------------- */

template <class DeviceType>
double MLIAPDummyModelKokkos<DeviceType>::memory_usage()
{
  // TODO: implement memory usage in Cython(?)
  return 0;
}

// not implemented
template <class DeviceType>
void MLIAPDummyModelKokkos<DeviceType>::read_coeffs(char *)
{
  MLIAPModelKokkos<DeviceType>::error->all(FLERR, "read_coeffs not implemented");
}

/* ----------------------------------------------------------------------
   build the unified interface object, connect to dummy model and descriptor
   ---------------------------------------------------------------------- */

template <class DeviceType>
MLIAPBuildUnifiedKokkos_t<DeviceType> LAMMPS_NS::build_unified(char *unified_fname, MLIAPDataKokkos<DeviceType> *data, LAMMPS *lmp,
                                             char *coefffilename)
{
  lmp->python->init();
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject *pyMain = PyImport_AddModule("__main__");

  if (!pyMain) {
    PyGILState_Release(gstate);
    lmp->error->all(FLERR, "Could not initialize embedded Python");
  }

  PyImport_ImportModule("mliap_unified_couple_kokkos");

  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
    PyGILState_Release(gstate);
    lmp->error->all(FLERR, "Loading mliappy unified module failure.");
  }

  // Connect dummy model, dummy descriptor, data to Python unified
  MLIAPDummyModelKokkos<DeviceType> *model = new MLIAPDummyModelKokkos<DeviceType>(lmp, coefffilename);
  MLIAPDummyDescriptorKokkos<DeviceType> *descriptor = new MLIAPDummyDescriptorKokkos<DeviceType>(lmp);

  PyObject *unified_interface = mliap_unified_connect_kokkos(unified_fname, model, descriptor);
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
    PyGILState_Release(gstate);
    lmp->error->all(FLERR, "Running mliappy unified module failure.");
  }

  // Borrowed references must be manually incremented
  model->unified_interface = unified_interface;
  Py_INCREF(unified_interface);
  descriptor->unified_interface = unified_interface;
  Py_INCREF(unified_interface);

  PyGILState_Release(gstate);

  MLIAPBuildUnifiedKokkos_t<DeviceType> build = {data, descriptor, model};
  return build;
}

/* ----------------------------------------------------------------------
   set energy for ij atom pairs
   ---------------------------------------------------------------------- */

void LAMMPS_NS::update_pair_energy(MLIAPDataKokkosDevice *data, double *eij)
{
  auto d_eatoms = data->eatoms;
  auto d_pair_i= data->pair_i;
  const auto nlocal = data->nlocal;
  Kokkos::parallel_for(nlocal, KOKKOS_LAMBDA(int ii) {
    d_eatoms[ii] = 0;
  });

  Kokkos::parallel_reduce(data->npairs, KOKKOS_LAMBDA(int ii, double &local_sum) {
    int i = d_pair_i[ii];
    double e = 0.5 * eij[ii];

    // must not count any contribution where i is not a local atom
    if (i < nlocal) {
      Kokkos::atomic_add(&d_eatoms[i], e);
      local_sum += e;
    }
  },*data->energy);
}

/* ----------------------------------------------------------------------
   set forces for ij atom pairs
   ---------------------------------------------------------------------- */

void LAMMPS_NS::update_pair_forces(MLIAPDataKokkosDevice *data, double *fij)
{
  auto *f = data->f;
  auto pair_i = data->pair_i;
  auto j_atoms = data->jatoms;
  auto vflag = data->vflag;
  auto rij = data->rij;
  int vflag_global = data->pairmliap->vflag_global;
  int vflag_atom = data->pairmliap->vflag_atom;
  if (vflag_atom) {
    data->pairmliap->k_vatom.modify_host();
    data->pairmliap->k_vatom.sync_device();
  }
  auto d_vatom = data->pairmliap->k_vatom.d_view;

  Kokkos::View<double[6], LMPDeviceType> virial("virial");

  Kokkos::parallel_for(data->npairs,KOKKOS_LAMBDA (int ii) {

    int ii3 = ii * 3;
    int i = pair_i[ii];
    int j = j_atoms[ii];
    // must not count any contribution where i is not a local atom
      Kokkos::atomic_add(&f[i*3+0], fij[ii3+0]);
      Kokkos::atomic_add(&f[i*3+1], fij[ii3+1]);
      Kokkos::atomic_add(&f[i*3+2], fij[ii3+2]);
      Kokkos::atomic_add(&f[j*3+0],-fij[ii3+0]);
      Kokkos::atomic_add(&f[j*3+1],-fij[ii3+1]);
      Kokkos::atomic_add(&f[j*3+2],-fij[ii3+2]);
      if (vflag) {
        double v[6];
        v[0] = -rij[ii3+0]*fij[ii3+0];
        v[1] = -rij[ii3+1]*fij[ii3+1];
        v[2] = -rij[ii3+2]*fij[ii3+2];
        v[3] = -rij[ii3+0]*fij[ii3+1];
        v[4] = -rij[ii3+0]*fij[ii3+2];
        v[5] = -rij[ii3+1]*fij[ii3+2];
        if (vflag_global) {
          Kokkos::atomic_add(&virial[0], v[0]);
          Kokkos::atomic_add(&virial[1], v[1]);
          Kokkos::atomic_add(&virial[2], v[2]);
          Kokkos::atomic_add(&virial[3], v[3]);
          Kokkos::atomic_add(&virial[4], v[4]);
          Kokkos::atomic_add(&virial[5], v[5]);
        }
        if (vflag_atom ) {
          Kokkos::atomic_add(&d_vatom(i,0), 0.5*v[0]);
          Kokkos::atomic_add(&d_vatom(i,1), 0.5*v[1]);
          Kokkos::atomic_add(&d_vatom(i,2), 0.5*v[2]);
          Kokkos::atomic_add(&d_vatom(i,3), 0.5*v[3]);
          Kokkos::atomic_add(&d_vatom(i,4), 0.5*v[4]);
          Kokkos::atomic_add(&d_vatom(i,5), 0.5*v[5]);

          Kokkos::atomic_add(&d_vatom(j,0), 0.5*v[0]);
          Kokkos::atomic_add(&d_vatom(j,1), 0.5*v[1]);
          Kokkos::atomic_add(&d_vatom(j,2), 0.5*v[2]);
          Kokkos::atomic_add(&d_vatom(j,3), 0.5*v[3]);
          Kokkos::atomic_add(&d_vatom(j,4), 0.5*v[4]);
          Kokkos::atomic_add(&d_vatom(j,5), 0.5*v[5]);
      }
    }
  });

  if (vflag) {
    if (vflag_global) {
      Kokkos::View<double[6], LMPHostType> h_virial("h_virial");
      Kokkos::deep_copy(h_virial,virial);
      for (int i = 0; i < 6; ++i)
        data->pairmliap->virial[i] += h_virial[i];
    }
    if (vflag_atom) {
      data->pairmliap->k_vatom.modify_device();
      data->pairmliap->k_vatom.sync_host();
    }
  }
}

/* ----------------------------------------------------------------------
   set energy for i indexed atoms
   ---------------------------------------------------------------------- */

void LAMMPS_NS::update_atom_energy(MLIAPDataKokkosDevice *data, double *ei)
{
  auto d_eatoms = data->eatoms;
  const auto nlocal = data->nlocal;

  Kokkos::parallel_reduce(nlocal, KOKKOS_LAMBDA(int i, double &local_sum) {
    double e = ei[i];

    d_eatoms[i] = e;
    local_sum += e;
  },*data->energy);
}

/* ----------------------------------------------------------------------
   set atom-wise forces
   ---------------------------------------------------------------------- */

void LAMMPS_NS::set_atom_forces(MLIAPDataKokkosDevice* data, double* Fi) {
  /*fprintf for Debugging*/
  // int n = data->nlocal < 3 ? data->nlocal : 3;
  // std::fprintf(stderr, "[DBG:set_f] nlocal=%d | |F_i|(first %d)=", data->nlocal, n);
  // for (int i=0;i<n;++i) {
  //   double fx=Fi[3*i+0], fy=Fi[3*i+1], fz=Fi[3*i+2];
  //   std::fprintf(stderr, " %.6g", std::sqrt(fx*fx+fy*fy+fz*fz));
  // }
  // std::fprintf(stderr, "\n"); 
  // std::fflush(stderr);

  // zero current forces
  Kokkos::parallel_for(data->ntotal*3, KOKKOS_LAMBDA(const int k){ data->f[k] = 0.0; });
  // write local atoms only
  Kokkos::parallel_for(data->nlocal, KOKKOS_LAMBDA(const int i){
    data->f[3*i+0] += Fi[3*i+0];
    data->f[3*i+1] += Fi[3*i+1];
    data->f[3*i+2] += Fi[3*i+2];
  });
  Kokkos::fence();
}

void LAMMPS_NS::set_total_energy(MLIAPDataKokkosDevice* data, double val) {
  /*fprintf for Debugging*/
  // std::fprintf(stderr, "[DBG:set_E] E_total=%.9g\n", val);
  // std::fflush(stderr);

  Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int){ *data->energy = val; });
  Kokkos::fence();
}

void LAMMPS_NS::add_global_virial(MLIAPDataKokkosDevice* data, const double* v6) {
  // pair class owns virial[6]; add to it on host side
  for (int k=0;k<6;++k) data->pairmliap->virial_model[k] += v6[k];
  data->pairmliap->has_model_virial = 1;

  /*fprintf for Debugging*/
  // std::fprintf(stderr, "[DBG:add_V] V6=[%.6g %.6g %.6g %.6g %.6g %.6g]\n",
  //              v6[0],v6[1],v6[2],v6[3],v6[4],v6[5]);
  // std::fflush(stderr);
}

static inline double det3(const double H[9]) {
  // row-major 3x3 determinant
  return  H[0]*(H[4]*H[8] - H[5]*H[7])
        - H[1]*(H[3]*H[8] - H[5]*H[6])
        + H[2]*(H[3]*H[7] - H[4]*H[6]);
}

void LAMMPS_NS::add_global_stress(MLIAPDataKokkosDevice *data, const double *s6_model)
{
  double S_lmp[6] = {
    s6_model[0], s6_model[1], s6_model[2],
    s6_model[3], s6_model[5], s6_model[4]
  };

  double V = det3(data->H);

  for (int k = 0; k < 6; ++k)
    data->pairmliap->virial_model[k] = S_lmp[k] * V;

  data->pairmliap->has_model_virial = 1;

  /*fprintf for Debugging*/
  // std::fprintf(stderr,
  //   "[DBG:add_S] S=[%.9g %.9g %.9g %.9g %.9g %.9g] V=%.9g -> vir=[%.9g %.9g %.9g %.9g %.9g %.9g]\n",
  //   s6_model[0], s6_model[1], s6_model[2], s6_model[3], s6_model[4], s6_model[5], V,
  //   data->pairmliap->virial_model[0], data->pairmliap->virial_model[1], data->pairmliap->virial_model[2],
  //   data->pairmliap->virial_model[3], data->pairmliap->virial_model[4], data->pairmliap->virial_model[5]);
  // std::fflush(stderr);
}

namespace LAMMPS_NS {
template class MLIAPDummyModelKokkos<LMPDeviceType>;
template class MLIAPDummyDescriptorKokkos<LMPDeviceType>;
template MLIAPBuildUnifiedKokkos_t<LMPDeviceType> LAMMPS_NS::build_unified(char *unified_fname, MLIAPDataKokkos<LMPDeviceType> *data, LAMMPS *lmp,
                                             char *coefffilename);
#ifdef LMP_KOKKOS_GPU
template class MLIAPDummyModelKokkos<LMPHostType>;
template class MLIAPDummyDescriptorKokkos<LMPHostType>;
template MLIAPBuildUnifiedKokkos_t<LMPHostType> LAMMPS_NS::build_unified(char *unified_fname, MLIAPDataKokkos<LMPHostType> *data, LAMMPS *lmp,
                                             char *coefffilename);
#endif
}
#endif

