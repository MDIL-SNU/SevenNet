/* Pair style e3gnn on the Apple GPU, see pair_e3gnn_mps.cpp.
   Part of SevenNet, MIT license (see LICENSE). It is compiled against LAMMPS headers:
   a LAMMPS build or plugin containing it is covered by the GPL of LAMMPS. */
#ifdef PAIR_CLASS
// clang-format off
PairStyle(e3gnn, PairE3GNNMps);
// clang-format on
#else

#ifndef LMP_PAIR_E3GNN_MPS_H
#define LMP_PAIR_E3GNN_MPS_H

#include <Python.h>

#include "pair.h"

namespace LAMMPS_NS {

class PairE3GNNMps : public Pair {
 public:
  explicit PairE3GNNMps(LAMMPS *lmp);
  ~PairE3GNNMps() override;
  void compute(int eflag, int vflag) override;
  void settings(int narg, char **arg) override;
  void coeff(int narg, char **arg) override;
  void init_style() override;
  double init_one(int i, int j) override;

 private:
  PyObject *driver = nullptr;
  double cutoff = 0.0;

  PyObject *import_driver();
  [[noreturn]] void fail(const char *what);
};

}  // namespace LAMMPS_NS

#endif
#endif
