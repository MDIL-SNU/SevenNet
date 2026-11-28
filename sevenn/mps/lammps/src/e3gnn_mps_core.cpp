/* Entry point of e3gnnmpscore.so, see e3gnn_mps_plugin.cpp.
   Part of SevenNet, MIT license (see LICENSE). It is compiled against LAMMPS headers:
   a LAMMPS build or plugin containing it is covered by the GPL of LAMMPS. */
#include "pair_e3gnn_mps.h"

extern "C" __attribute__((visibility("default"))) void *e3gnn_mps_create(void *lmp) {
  return new LAMMPS_NS::PairE3GNNMps(static_cast<LAMMPS_NS::LAMMPS *>(lmp));
}
