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
PairStyle(e3gnn/parallel,PairE3GNNParallel)

#else

#ifndef LMP_PAIR_E3GNN_PARALLEL
#define LMP_PAIR_E3GNN_PARALLEL

#include "pair.h"

#include <torch/torch.h>
#include <vector>

namespace LAMMPS_NS{
  class PairE3GNNParallel : public Pair{
    private:
      double cutoff;
      double cutoff_square;
      float shift;
      float scale;
      std::vector<torch::jit::Module> model_list;
      torch::Device device = torch::kCPU;
      torch::Device get_cuda_device();

      // for communication
      int graph_size;
      int x_dim; //to determine per atom data size
      torch::Tensor x_comm; // x_local + x_ghost + x_comm_extra

      bool comm_preprocess_done = false;
      void comm_preprocess();
      void pack_forward_init(int comm_phase, int n, int *list);
      void unpack_forward_init(int comm_phase, int n, int first, double* buf_check);

      // temporary variable holds for each compute step
      std::unordered_map<int, int> extra_graph_idx_map;
      std::vector<int> comm_index_pack_forward[6];
      std::vector<int> comm_index_unpack_forward[6];

      // pointer to buf for comm. used for check self communication
      double* buf_hold;
      // to use tag_to_graph_idx inside comm methods
      int* tag_to_graph_idx_ptr=nullptr;

    public:
      PairE3GNNParallel(class LAMMPS *);
      ~PairE3GNNParallel();
      void compute(int, int) override;

      void settings(int, char **) override;
      //read Atom type string from input script & related coeff
      void coeff(int, char **) override;
      void allocate();

      int pack_forward_comm(int, int *, double *, int, int *) override;
      void unpack_forward_comm(int, int, double *) override;
      int pack_reverse_comm(int, int, double *) override;
      void unpack_reverse_comm(int, int *, double *) override;

      void init_style() override;
      double init_one(int, int) override;
  };
}

#endif
#endif
