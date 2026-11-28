/* ----------------------------------------------------------------------
   Pair style e3gnn for the Apple GPU: plain `lmp -in`, serial.

   As the NVIDIA pair style, with the checkpoint (or a pretrained name) in place of the deployed model:
       pair_style  e3gnn
       pair_coeff  * * checkpoint.pth [keywords] O Zr

   TorchScript cannot hold the Metal kernels, so this embeds Python and calls
   sevenn/mps/lammps/pair.py, which owns the model. Build with `sevenn patch_lammps` on a Mac, or build_plugin.sh.

   Part of SevenNet, MIT license (see LICENSE). It is compiled against LAMMPS headers:
   a LAMMPS build or plugin containing it is covered by the GPL of LAMMPS.
------------------------------------------------------------------------- */

#include <Python.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "pair_e3gnn_mps.h"
#include "update.h"

using namespace LAMMPS_NS;

namespace {

class PythonLock {
 public:
  PythonLock() : state(PyGILState_Ensure()) {}
  ~PythonLock() { PyGILState_Release(state); }

 private:
  PyGILState_STATE state;
};

// Inside the environment the plugin was built for. Stays up until exit: torch does not survive Py_Finalize.
void start_python(Error *error) {
  if (Py_IsInitialized()) return;
  const char *env = getenv("SEVENN_PYTHON");
  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  config.install_signal_handlers = 0;
  PyStatus status = PyConfig_SetBytesString(&config, &config.program_name, env ? env : SEVENN_DEFAULT_PYTHON);
  if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status)) error->one(FLERR, std::string("e3gnn: cannot start Python: ") + (status.err_msg ? status.err_msg : "unknown"));
  PyEval_SaveThread();
}

PyObject *box_tuple(Domain *domain) {
  return Py_BuildValue("(ddd)(ddd)ddd(iii)", domain->boxlo[0], domain->boxlo[1], domain->boxlo[2], domain->boxhi[0],
                       domain->boxhi[1], domain->boxhi[2], domain->xy, domain->xz, domain->yz, domain->xperiodic,
                       domain->yperiodic, domain->zperiodic);
}

PyObject *view(void *data, Py_ssize_t bytes, int flags) {
  return PyMemoryView_FromMemory(static_cast<char *>(data), bytes, flags);
}

}  // namespace

PairE3GNNMps::PairE3GNNMps(LAMMPS *lmp) : Pair(lmp) {
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
}

PairE3GNNMps::~PairE3GNNMps() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  if (driver && Py_IsInitialized()) {
    PythonLock lock;
    Py_DECREF(driver);
  }
}

void PairE3GNNMps::settings(int narg, char **) {
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");
}

void PairE3GNNMps::coeff(int narg, char **arg) {
  if (narg < 3 || strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "e3gnn: expected pair_coeff * * checkpoint.pth [keywords] element1 element2 ...");
  if (allocated) error->all(FLERR, "e3gnn: pair_coeff called twice");
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  start_python(error);
  PythonLock lock;
  PyObject *args = PyList_New(narg - 2);
  for (int i = 2; args && i < narg; i++) {
    PyObject *word = PyUnicode_FromString(arg[i]);
    if (!word) {
      Py_CLEAR(args);
      break;
    }
    PyList_SET_ITEM(args, i - 2, word);  // steals the reference
  }
  PyObject *box = args ? box_tuple(domain) : nullptr;
  PyObject *module = box ? import_driver() : nullptr;
  PyObject *made = module ? PyObject_CallMethod(module, "PairDriver", "OiO", args, n, box) : nullptr;
  Py_XDECREF(module);
  Py_XDECREF(box);
  Py_XDECREF(args);
  if (!made) fail("loading the model");
  driver = made;

  PyObject *cut = PyObject_GetAttrString(driver, "cutoff");
  PyObject *banner = PyObject_GetAttrString(driver, "banner");
  cutoff = cut ? PyFloat_AsDouble(cut) : -1.0;
  const char *text = banner ? PyUnicode_AsUTF8(banner) : nullptr;
  if (!cut || !banner || !text || PyErr_Occurred()) {
    Py_XDECREF(cut);
    Py_XDECREF(banner);
    fail("reading the model");
  }
  utils::logmesg(lmp, "{}\n", text);
  Py_DECREF(cut);
  Py_DECREF(banner);

  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutoff * cutoff;
    }
}

// The model builds its own edges from the local atoms, so no neighbor list and no ghost atoms are used.
void PairE3GNNMps::init_style() {
  if (comm->nprocs > 1) error->all(FLERR, "e3gnn on the Apple GPU is serial: run LAMMPS on one MPI rank");
  if (strcmp(update->unit_style, "metal") != 0)
    error->all(FLERR, std::string("e3gnn needs units metal, got ") + update->unit_style);
  if (neighbor->nex_type || neighbor->nex_group || neighbor->nex_mol) error->all(FLERR, "e3gnn builds its own edges and cannot honour neigh_modify exclude");
  if (neighbor->includegroup) error->all(FLERR, "e3gnn builds its own edges and cannot honour neigh_modify include");
  if (force->pair != this) error->all(FLERR, "e3gnn on the Apple GPU takes every atom: it cannot be a sub-style of pair_style hybrid");
  if (atom->natoms == 0) error->all(FLERR, "e3gnn needs at least one atom");
}

double PairE3GNNMps::init_one(int, int) { return cutoff; }

void PairE3GNNMps::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  if (vflag_atom) error->all(FLERR, "e3gnn: atomic stress is not supported");
  const int nlocal = atom->nlocal;
  if (nlocal == 0) return;  // all atoms lost, with thermo_modify lost ignore
  double virial_now[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double energy = 0.0;
  {
    PythonLock lock;
    PyObject *x = view(atom->x[0], 24L * nlocal, PyBUF_READ);
    PyObject *f = view(atom->f[0], 24L * nlocal, PyBUF_WRITE);
    PyObject *type = view(atom->type, 4L * nlocal, PyBUF_READ);
    PyObject *eper = eflag_atom ? view(eatom, 8L * nlocal, PyBUF_WRITE) : (Py_INCREF(Py_None), Py_None);
    PyObject *vir = view(virial_now, sizeof(virial_now), PyBUF_WRITE);
    PyObject *box = box_tuple(domain);
    PyObject *out = nullptr;
    if (x && f && type && eper && vir && box)
      out = PyObject_CallMethod(driver, "compute", "iOOOOOO", nlocal, x, f, type, eper, vir, box);
    for (PyObject *o : {x, f, type, eper, vir, box}) Py_XDECREF(o);
    if (!out) fail("computing forces");
    energy = PyFloat_AsDouble(out);
    Py_DECREF(out);
    if (PyErr_Occurred()) fail("reading the energy");
  }
  if (eflag_global) eng_vdwl += energy;
  if (vflag_global)
    for (int k = 0; k < 6; k++) virial[k] += virial_now[k];
}

PyObject *PairE3GNNMps::import_driver() {
  PyObject *sys_path = PySys_GetObject("path");
  const char *root = getenv("SEVENN_ROOT");
  PyObject *where = PyUnicode_FromString(root ? root : SEVENN_DEFAULT_ROOT);
  if (!sys_path || !where) {
    Py_XDECREF(where);
    return nullptr;
  }
  const int present = PySequence_Contains(sys_path, where);
  if (present < 0 || (present == 0 && PyList_Insert(sys_path, 0, where) < 0)) {
    Py_DECREF(where);
    return nullptr;
  }
  Py_DECREF(where);
  return PyImport_ImportModule("sevenn.mps.lammps.pair");
}

void PairE3GNNMps::fail(const char *what) {
  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);
  std::string text = "unknown";
  PyObject *str = value ? PyObject_Str(value) : nullptr;
  if (str) {
    const char *utf8 = PyUnicode_AsUTF8(str);
    if (utf8) text = utf8;
    Py_DECREF(str);
  }
  PyErr_Clear();  // a failure while formatting must not hide the original error
  PyErr_Display(type, value, traceback);
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  fflush(stderr);
  error->one(FLERR, std::string("e3gnn: Python error while ") + what + ": " + text);
}
