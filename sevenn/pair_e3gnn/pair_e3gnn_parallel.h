/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
http://lammps.sandia.gov, Sandia National Laboratories
Steve Plimpton, sjplimp@sandia.gov

Copyright (2003) Sandia Corporation.  Under the terms of Contract
DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
certain rights in this software.  This software is distributed under
the GNU General Public License.

See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
PairStyle(e3gnn/parallel, PairE3GNNParallel)

#else

#ifndef LMP_PAIR_E3GNN_PARALLEL
#define LMP_PAIR_E3GNN_PARALLEL

#include "pair.h"

#include <torch/torch.h>
#include <vector>

namespace LAMMPS_NS {
class PairE3GNNParallel : public Pair {
private:
  double cutoff;
  double cutoff_square;
  std::vector<torch::jit::Module> model_list;
  torch::Device device = torch::kCPU;
  torch::Device device_comm = torch::kCPU;
  torch::Device get_cuda_device();
  bool use_cuda_mpi;

  // for communication
  // Most of these variables for communication is temporary and valid for only
  // one MD step.
  int x_dim; // to determine per atom data size
  int graph_size;
  torch::Tensor x_comm; // x_local + x_ghost + x_comm_extra

  void comm_preprocess();
  bool comm_preprocess_done = false;

  // temporary variables holds for each compute step
  std::unordered_map<int, long> extra_graph_idx_map;
  // To use scatter, store long instead of int
  // array of vector
  std::vector<long> comm_index_pack_forward[6];
  std::vector<long> comm_index_unpack_forward[6];
  std::vector<long> comm_index_unpack_reverse[6];

  // its size is 6 and initialized at comm_preprocess()
  torch::Tensor comm_index_pack_forward_tensor[6];
  torch::Tensor comm_index_unpack_forward_tensor[6];
  torch::Tensor comm_index_unpack_reverse_tensor[6];

  // to use tag_to_graph_idx inside comm methods
  int *tag_to_graph_idx_ptr = nullptr;

  int sendproc[6];
  int recvproc[6];

public:
  PairE3GNNParallel(class LAMMPS *);
  ~PairE3GNNParallel();

  // TODO: keep encapsulation..
  void compute(int, int) override;
  void settings(int, char **) override;
  // read Atom type string from input script & related coeff
  void coeff(int, char **) override;
  void allocate();

  void pack_forward_init(int n, int *list, int comm_phase);
  void unpack_forward_init(int n, int first, int comm_phase);

  int pack_forward_comm_gnn(float *buf, int comm_phase);
  void unpack_forward_comm_gnn(float *buf, int comm_phase);
  int pack_reverse_comm_gnn(float *buf, int comm_phase);
  void unpack_reverse_comm_gnn(float *buf, int comm_phase);

  void init_style() override;
  double init_one(int, int) override;

  int get_x_dim();
  bool use_cuda_mpi_();
  bool is_comm_preprocess_done();
  void notify_proc_ids(const int *sendproc, const int *recvproc);

  bool print_info = false;
  int world_rank;
};

class DeviceBuffManager {
private:
  DeviceBuffManager() {}
  DeviceBuffManager(const DeviceBuffManager &);
  DeviceBuffManager &operator=(const DeviceBuffManager &);

  float *buf_send_device = nullptr;
  float *buf_recv_device = nullptr;
  int send_buf_size = 0;
  int recv_buf_size = 0;

public:
  static DeviceBuffManager &getInstance();
  void get_buffer(int, int, float *&, float *&);

  ~DeviceBuffManager();
};
} // namespace LAMMPS_NS

#endif
#endif
