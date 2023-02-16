/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Yutack Park (SNU)
------------------------------------------------------------------------- */

#include <ATen/ops/from_blob.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <string>

#include <torch/torch.h>
#include <torch/script.h>

#include "memory.h"
#include "error.h"
#include "atom.h"
#include "force.h"
#include "neighbor.h" 
#include "neigh_list.h"
#include "neigh_request.h"

#include "pair_e3gnn.h"

using namespace LAMMPS_NS;

PairE3GNN::PairE3GNN(LAMMPS *lmp) : Pair(lmp) {
  // constructor
  std::string device_name;
  if(torch::cuda::is_available()){
    device = torch::kCUDA;
    device_name = "CUDA";
  }
  else {
    device = torch::kCPU;
    device_name = "CPU";
  }

  if (lmp->logfile) {
    fprintf(lmp->logfile, "PairE3GNN using device : %s\n", device_name.c_str());
  }
}

PairE3GNN::~PairE3GNN() {
  if(allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(map);
    memory->destroy(elements);
  }
}

void PairE3GNN::compute(int eflag, int vflag) {
  //compute
  /*
     read 
     https://pytorch.org/cppdocs/notes/tensor_basics.html
     problems :
     edge_length is already calculated here. (what about model?)
     using tag -> what happen in mpi mode?
     calculation in cpu side (using accessor) is it good?
  */
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;
  if(vflag_atom) {
    error->all(FLERR,"atomic stress related feature is not supported\n");
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = list->inum;  // same as nlocal
  long num_atoms[1] = {nlocal};
  int* ilist = list->ilist;

  tagint *tag = atom->tag;
  int tag2i[nlocal];

  int* numneigh = list->numneigh;  // j loop cond
  int** firstneigh = list->firstneigh;  // j list
  const int nedges_upper_bound = std::accumulate(numneigh, numneigh+nlocal, 0);

  //int nedges = std::accumulate(numneigh, numneigh+ntotal, 0);

  torch::Tensor inp_node_type = torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  auto inp_num_atoms = torch::from_blob(num_atoms, {1},torch::TensorOptions().dtype(torch::kInt64));

  // use global index given by ilist (or jlist) -> is it right way to do?
  // it will cause almost random access pattern 
  auto node_type = inp_node_type.accessor<long, 1>();
  float edge_vec_tmp[nedges_upper_bound][3];
  long edge_idx_src[nedges_upper_bound];
  long edge_idx_dst[nedges_upper_bound];

  int nedges = 0;

  for (int ii = 0; ii < nlocal; ii++) {
    const int i = ilist[ii];
    int itag = tag[i];
    tag2i[itag-1] = i;
    const int itype = type[i];
    const int* jlist = firstneigh[i];
    const int jnum = numneigh[i];
    node_type[itag-1] = map[itype];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      int jtag = tag[j];
      j &= NEIGHMASK;
      const int jtype = type[j];

      // we have to calculate Rij to check cutoff in lammps side
      const double delij[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1], x[j][2] - x[i][2]};
      const double Rij = delij[0]*delij[0] + delij[1]*delij[1] + delij[2]*delij[2];
      if(Rij < cutoff_square) {
        edge_idx_src[nedges] = itag-1;
        edge_idx_dst[nedges] = jtag-1;
        edge_vec_tmp[nedges][0] = delij[0];
        edge_vec_tmp[nedges][1] = delij[1];
        edge_vec_tmp[nedges][2] = delij[2];
        // add edge_length?
        //edge_len[nedges] = Rij;
        nedges++;
      }
    } // j loop end
  } // i loop end

  // way to init 2, nedges from double array? (without src, dst)
  auto edge_idx_src_tensor = torch::from_blob(edge_idx_src, {nedges},torch::TensorOptions().dtype(torch::kInt64));
  auto edge_idx_dst_tensor = torch::from_blob(edge_idx_dst, {nedges},torch::TensorOptions().dtype(torch::kInt64));
  auto inp_edge_index = torch::stack({edge_idx_src_tensor, edge_idx_dst_tensor});

  auto inp_edge_vec = torch::from_blob(edge_vec_tmp, {nedges, 3});
  //torch::Tensor inp_edge_len = torch::zeros({nedges});

  //auto edge_len = inp_edge_len.accessor<float, 1>();

  // should be part of model?
  inp_edge_vec.set_requires_grad(true);

  c10::Dict<std::string, torch::Tensor> input_dict;
  input_dict.insert("x", inp_node_type.to(device));
  input_dict.insert("edge_index", inp_edge_index.to(device));
  input_dict.insert("edge_vec", inp_edge_vec.to(device));
  input_dict.insert("num_atoms", inp_num_atoms.to(device));

  /*
  std::cout << inp_node_type << "\n";
  std::cout << inp_edge_index << "\n";
  std::cout << inp_edge_vec << "\n";
  */

  std::vector<torch::IValue> input(1, input_dict);
  auto output = model.forward(input).toGenericDict();

  // atomic energy things?
  torch::Tensor total_energy_tensor = output.at("inferred_total_energy").toTensor().cpu();
  torch::Tensor force_tensor = output.at("inferred_force").toTensor().cpu();
  auto forces = force_tensor.accessor<float, 2>();
  //std::cout << total_energy_tensor << '\n';
  //std::cout << force_tensor << '\n';
  eng_vdwl += total_energy_tensor.item<float>();

  for(int itag = 0; itag < nlocal; itag++){
    int i = tag2i[itag];
    f[i][0] = forces[itag][0];
    f[i][1] = forces[itag][1];
    f[i][2] = forces[itag][2];
  }

  if (vflag_fdotr) virial_fdotr_compute(); // is it safe to use? pressure calc
}

// allocate arrays (called from coeff)
void PairE3GNN::allocate() {
  allocated = 1;
  int n = atom->ntypes;
  
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(map,n+1,"pair:map");
}

// global settings for pair_style
void PairE3GNN::settings(int narg, char **arg) {
  if (narg != 0) {
    error->all(FLERR,"Illegal pair_style command");
  }
}

// set coeff of types from pair_coeff
// TODO: this function is temporary!, initiaize torch performance things, type_map, etc
void PairE3GNN::coeff(int narg, char **arg) {
  ///////////////////////////////////////////
  /*
     here, from user input like 'Hf O' and lammps interface the type number
     is given. and using the metadata of model we have to make function
     that type number -> one hot atom ebedding index
     note that lammps type number start from 1
     no matter what sevenn python does, here all we need is function that
     'Hf O' -> one hot atom ebedding index
     the original pair_nn scheme is make map : type_number -> element
     than from compute, atom_index -> type (atom -> type) to get element
 */
  if(allocated) {
    error->all(FLERR,"pair_e3gnn coeff called twice");
  }
  allocate();

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0) {
      error->all(FLERR, "e3gnn: firt and second input of pair_coeff should be '*'");
  }
  // expected input : pair_coeff * * pot.pth type_name1 type_name2 ...

  std::unordered_map<std::string, std::string> meta_dict = {
    {"chemical_symbols_to_index", ""},
    {"cutoff", ""},
    {"num_species", ""},
    {"model_type", ""},
    {"version", ""},
    {"dtype", ""},
    {"time", ""}
  };

  // model loading from input
  model = torch::jit::load(std::string(arg[2]), device, meta_dict);
  //model = torch::jit::freeze(model); model is already freezed

  torch::jit::FusionStrategy strategy;
  // thing about dynamic recompile as tensor shape varies, this is default
  strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}}; 
  torch::jit::setFusionStrategy(strategy);

  cutoff = std::stod(meta_dict["cutoff"]);
  cutoff_square = cutoff * cutoff;

  if(meta_dict["model_type"].compare("E3_equivariant_model") != 0){
    error->all(FLERR, "given model type is not E3_equivariant_model");    
  }

  std::string chem_str = meta_dict["chemical_symbols_to_index"];
  int ntypes = atom->ntypes;

  auto delim = " ";
  char *tok = std::strtok(const_cast<char*>(chem_str.c_str()), delim);
  std::vector<std::string> chem_vec;
  while (tok != nullptr){
    chem_vec.push_back(std::string(tok));
    tok = std::strtok(nullptr, delim);
  }

  // what if unkown chemical specie is in arg? should I abort? is there any use case for that?
  for (int i=3; i<narg; i++) {
    for (int j=0; j<chem_vec.size(); j++) {
      if (chem_vec[j].compare(arg[i]) == 0) {
        map[i-2] = j;
      }
    }
  }

  for (int i = 1; i <= ntypes; i++) {
    for (int j = 1; j <= ntypes; j++) {
      if ((map[i] >= 0) && (map[j] >= 0)) {
          setflag[i][j] = 1;
          cutsq[i][j] = cutoff*cutoff;
      }
    }
  }

  if (lmp->logfile) {
    fprintf(lmp->logfile, "from sevenn version '%s' ", meta_dict["version"].c_str());
    fprintf(lmp->logfile, "%s precision model trained at %s is loaded\n", meta_dict["dtype"].c_str(), meta_dict["time"].c_str());
  }

}

// init specific to this pair
void PairE3GNN::init_style() {
  // parallel version ?
  /*
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style nn requires newton pair on");    
  }
  */

  // full neighbor list of course (this is many-body potential)
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairE3GNN::init_one(int i, int j) {
  return cutoff;
}
