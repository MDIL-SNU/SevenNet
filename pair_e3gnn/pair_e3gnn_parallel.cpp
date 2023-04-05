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

#include <ATen/core/Dict.h>
#include <ATen/core/ivalue_inl.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <numeric>
#include <string>

#include <torch/csrc/jit/api/module.h>
#include <torch/torch.h>
#include <torch/script.h>

#include "memory.h"
#include "error.h"
#include "atom.h"
#include "force.h"
#include "neighbor.h" 
#include "neigh_list.h"
#include "neigh_request.h"
#include "comm.h"
#include "error.h"

#include "pair_e3gnn_parallel.h"

using namespace LAMMPS_NS;

#define INTEGER_TYPE torch::TensorOptions().dtype(torch::kInt64)
#define FLOAT_TYPE torch::TensorOptions().dtype(torch::kFloat)

PairE3GNNParallel::PairE3GNNParallel(LAMMPS *lmp) : Pair(lmp) {
  // constructor
  std::string device_name;
  if(torch::cuda::is_available()){
    device = get_cuda_device();
    device_name = "CUDA";
  } else {
    device = torch::kCPU;
    device_name = "CPU";
  }

  comm_forward = 0;
  comm_reverse = 0;
  x_comm_hold = nullptr;
  nmax = 0;

  if (lmp->logfile) {
    fprintf(lmp->logfile, "PairE3GNNParallel using device : %s\n", device_name.c_str());
  }
}

torch::Device PairE3GNNParallel::get_cuda_device() {
  char* cuda_visible = std::getenv("CUDA_VISIBLE_DEVICES");
  int num_gpus;
  int idx;
  int rank = comm->me;
  if(cuda_visible == nullptr){
    // assume every gpu in node is avail
    num_gpus = torch::cuda::device_count();
    //believe user did right thing...
    idx = rank % num_gpus;
  } else {
    auto delim = ",";
    char *tok = std::strtok(cuda_visible, delim);
    std::vector<std::string> device_ids;
    while(tok != nullptr) {
      device_ids.push_back(std::string(tok));
      tok = std::strtok(nullptr, delim);
    }
    idx = std::stoi(device_ids[rank % device_ids.size()]);
  }
  return torch::Device(torch::kCUDA, idx);
}

PairE3GNNParallel::~PairE3GNNParallel() {
  if(allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(map);
    memory->destroy(x_comm_hold);
  }
}

void PairE3GNNParallel::compute(int eflag, int vflag) {
  /*
     read 
     https://pytorch.org/cppdocs/notes/tensor_basics.html
     problems :
     calculation in cpu side (using accessor) is it good?

     map vs large sparse array (x_comm_hold, tag_to_graph_idx)
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
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int* ilist = list->ilist;
  int inum = list->inum;

  //nice approach from pair_eam
  if (atom->nmax > nmax) {
    memory->destroy(x_comm_hold);
    nmax = atom->nmax;
    memory->create(x_comm_hold, nmax, comm_forward, "PairE3GNNParallel:x_comm_hold");
  }

  bigint natoms = atom->natoms;
  tagint *tag = atom->tag;
  int tag_to_graph_idx[natoms+1];  // this should be smaller than ntotal
  std::fill_n(tag_to_graph_idx, natoms+1, -1);
  tag_to_graph_idx_ptr = tag_to_graph_idx;
  int graph_indexer = nlocal;
  int graph_index_to_i[ntotal];

  int* numneigh = list->numneigh;  // j loop cond
  int** firstneigh = list->firstneigh;  // j list
  const int nedges_upper_bound = std::accumulate(numneigh, numneigh+nlocal, 0);

  std::vector<long> node_type;
  std::vector<long> node_type_ghost;

  float edge_vec[nedges_upper_bound][3];
  long edge_idx_src[nedges_upper_bound];
  long edge_idx_dst[nedges_upper_bound];

  int nedges = 0;
  for (int ii = 0; ii < inum; ii++) {
    // populate tag_to_graph_idx first
    const int i = ilist[ii];
    int itag = tag[i];
    const int itype = type[i];
    tag_to_graph_idx[itag] = ii;
    graph_index_to_i[ii] = i;
    node_type.push_back(map[itype]);
  }

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    //int itag = tag[i];
    const int i_graph_idx = ii;
    const int itype = type[i];
    const int* jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      int jtag = tag[j];
      j &= NEIGHMASK;
      const int jtype = type[j];
      // we have to calculate Rij to check cutoff in lammps side
      const double delij[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1], x[j][2] - x[i][2]};
      const double Rij = delij[0]*delij[0] + delij[1]*delij[1] + delij[2]*delij[2];

      int j_graph_idx;
      if(Rij < cutoff_square) {
        if(tag_to_graph_idx[jtag] == -1) {
          tag_to_graph_idx[jtag] = graph_indexer;
          graph_index_to_i[graph_indexer] = j;
          node_type_ghost.push_back(map[jtype]);
          graph_indexer++;
        }

        j_graph_idx = tag_to_graph_idx[jtag];
        edge_idx_src[nedges] = i_graph_idx;
        edge_idx_dst[nedges] = j_graph_idx;
        edge_vec[nedges][0] = delij[0];
        edge_vec[nedges][1] = delij[1];
        edge_vec[nedges][2] = delij[2];
        nedges++;
      }
    } // j loop end
  } // i loop end
  const int ghost_node_num = graph_indexer - nlocal;

  // convert primitive data to torch Tensor
  auto inp_node_type = torch::from_blob(node_type.data(), nlocal, INTEGER_TYPE);
  auto inp_node_type_ghost = torch::from_blob(node_type_ghost.data(), ghost_node_num, INTEGER_TYPE);

  long num_nodes[1] = {long(nlocal)};
  auto inp_num_atoms = torch::from_blob(num_nodes, {1}, INTEGER_TYPE);

  auto edge_idx_src_tensor = torch::from_blob(edge_idx_src, {nedges}, INTEGER_TYPE);
  auto edge_idx_dst_tensor = torch::from_blob(edge_idx_dst, {nedges}, INTEGER_TYPE);
  auto inp_edge_index = torch::stack({edge_idx_src_tensor, edge_idx_dst_tensor});

  auto inp_edge_vec = torch::from_blob(edge_vec, {nedges, 3}, FLOAT_TYPE);

  // r_original requires grad True
  inp_edge_vec.set_requires_grad(true);

  torch::Dict<std::string, torch::Tensor> input_dict;
  input_dict.insert("x", inp_node_type.to(device));
  input_dict.insert("x_ghost", inp_node_type_ghost.to(device));
  input_dict.insert("edge_index", inp_edge_index.to(device));
  input_dict.insert("edge_vec", inp_edge_vec.to(device));
  input_dict.insert("num_atoms", inp_num_atoms.to(device));
  input_dict.insert("nlocal", inp_num_atoms.to(torch::kCPU));

  //TODO: edge attr, edge embedding
  std::list<std::vector<torch::Tensor>> wrt_tensors;
  wrt_tensors.push_back({input_dict.at("edge_vec")});

  auto model_part = model_list.front();
  auto output = model_part.forward({input_dict}).toGenericDict();

  for(auto it = model_list.begin(); it != model_list.end(); ++it) {
    if(it == model_list.begin()) continue;
    model_part = *it;

    x_local = output.at("x").toTensor().to(torch::kCPU);

    x_dim = x_local.size(1); // size of comm for each atom
    x_ghost = torch::zeros({ghost_node_num, x_dim}); //by default, in CPU

    comm->forward_comm(this); //populate x_ghost by communication

    x_ghost.set_requires_grad(true);
    output.insert_or_assign("x_ghost", x_ghost.to(device));

    output.insert_or_assign("edge_vec", output.at("edge_vec").toTensor().clone());

    wrt_tensors.push_back({output.at("edge_vec").toTensor(), \
                           output.at("x").toTensor(), \
                           output.at("self_cont_tmp").toTensor(), \
                           output.at("x_ghost").toTensor()});

    output = model_part.forward({output}).toGenericDict();
  }
  // {1, 1} to {1}
  torch::Tensor scaled_energy_tensor = output.at("scaled_total_energy").toTensor().squeeze();
  eng_vdwl += scaled_energy_tensor.item<float>()*scale + shift*nlocal; // accumulate energy, fix later

  // zero out x_comm_hold
  for(int l=0; l<nmax; l++) {
    for(int m=0; m<x_dim; m++) {
      x_comm_hold[l][m] = 0;
    }
  }

  x_local = torch::ones({1}); // first grad_output (dE_dE = 1)
  torch::Tensor self_conn_grads;
  torch::Tensor dE_dr = torch::zeros({nedges, 3}, FLOAT_TYPE);
  torch::Tensor x_local_save; //holds grad info of x_local (it loses its grad when sends to CPU)
  std::vector<torch::Tensor> grads;
  std::vector<torch::Tensor> of_tensor;

  // most values of self_conn_grads are zero becuase we use only scalars for energy
  for(auto rit = wrt_tensors.rbegin(); rit != wrt_tensors.rend(); ++rit) {
    // edge_vec, x, x_ghost order
    auto wrt_tensor = *rit;
    if(rit == wrt_tensors.rbegin()) {
      grads = torch::autograd::grad({scaled_energy_tensor}, wrt_tensor);
    } else {
      x_local_save.copy_(x_local);
      grads = torch::autograd::grad(of_tensor, wrt_tensor, {x_local_save, self_conn_grads});
    }
    dE_dr = dE_dr + grads.at(0).to(torch::kCPU); //accumulate force
    if(std::distance(rit, wrt_tensors.rend()) == 1) continue;  // if last iteration

    of_tensor.clear();
    of_tensor.push_back(wrt_tensor[1]);
    of_tensor.push_back(wrt_tensor[2]);

    x_local_save = grads.at(1);  // device location
    x_local = grads.at(1).to(torch::kCPU);  // grad_outputs (dE_dx)
    self_conn_grads = grads.at(2);
    x_ghost = grads.at(3).to(torch::kCPU);  // store dE_dx_ghost

    x_dim = x_local.size(1);
    comm->reverse_comm(this);  // comm dE_dx_ghost to other proc to complete dE_dx
    // now x_local is complete (dE_dx)

    // zero out x_comm_hold
    for(int l=0; l<nmax; l++) {
      for(int m=0; m<x_dim; m++) {
        x_comm_hold[l][m] = 0;
      }
    }
  }

  // atomic energy things?
  torch::Tensor force_tensor = torch::zeros({graph_indexer, 3});

  // TODO:where I can find torch_scatter cpp version? I heard this version(using built) is slower.
  force_tensor.scatter_(0, edge_idx_src_tensor.repeat_interleave(3).view({nedges,3}), dE_dr, "add");
  force_tensor.scatter_(0, edge_idx_dst_tensor.repeat_interleave(3).view({nedges,3}), torch::neg(dE_dr), "add");

  force_tensor = force_tensor.mul(scale);
  auto forces = force_tensor.accessor<float, 2>();

  for (int graph_idx=0; graph_idx < graph_indexer; graph_idx++){
    int i = graph_index_to_i[graph_idx];
    f[i][0] = forces[graph_idx][0];
    f[i][1] = forces[graph_idx][1];
    f[i][2] = forces[graph_idx][2];
  }
}

// allocate arrays (called from coeff)
void PairE3GNNParallel::allocate() {
  allocated = 1;
  int n = atom->ntypes;
  
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(map,n+1,"pair:map");
}

// global settings for pair_style
void PairE3GNNParallel::settings(int narg, char **arg) {
  if (narg != 0) {
    error->all(FLERR,"Illegal pair_style command");
  }
}

void PairE3GNNParallel::coeff(int narg, char **arg) {
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
    {"time", ""},
    {"shift", ""},
    {"scale", ""},
    {"comm_size", ""}
  };

  // model loading from input
  int n_model = std::stoi(arg[2]);
  for (int i=3; i<n_model+3; i++){
    model_list.push_back(torch::jit::load(std::string(arg[i]), device, meta_dict));
  }

  torch::jit::FusionStrategy strategy;
  // TODO: why first pew iteration is slower than nequip?
  strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}}; 
  torch::jit::setFusionStrategy(strategy);

  cutoff = std::stod(meta_dict["cutoff"]);
  shift = std::stod(meta_dict["shift"]);
  scale = std::stod(meta_dict["scale"]);

  int comm_size = std::stod(meta_dict["comm_size"]);
  // to initialize buffer size for communication
  comm_forward = comm_size;  //variable of parent class
  comm_reverse = comm_size;

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
  for (int i=3+n_model; i<narg; i++) {
    for (int j=0; j<chem_vec.size(); j++) {
      if (chem_vec[j].compare(arg[i]) == 0) {
        map[i-2-n_model] = j; //store from 1, (not 0)
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
    fprintf(lmp->logfile, "from simple_gnn version '%s' ", meta_dict["version"].c_str());
    fprintf(lmp->logfile, "%s precision model trained at %s is loaded\n", meta_dict["dtype"].c_str(), meta_dict["time"].c_str());
  }

}

// init specific to this pair
void PairE3GNNParallel::init_style() {
  // parallel version ?
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style nn requires newton pair on");    
  }

  // full neighbor list of course (this is many-body potential)
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairE3GNNParallel::init_one(int i, int j) {
  return cutoff;
}

// TODO: refactor for speed
int PairE3GNNParallel::pack_forward_comm(int n, int *list_send, double *buf, int pbc_flag, int* pbc) {
  int i,j,m;
  int nlocal = list->inum;
  tagint *tag = atom->tag;
  auto x_local_accessor = x_local.accessor<float, 2>();

  m = 0;
  for (i = 0; i < n; i++) {
    int list_i = list_send[i];
    int graph_idx = tag_to_graph_idx_ptr[tag[list_i]];
    if(graph_idx < nlocal && graph_idx != -1) {
      for (j = 0; j < x_dim; j++) {
        buf[m++] = static_cast<double>(x_local_accessor[graph_idx][j]);
      }
    } else {
      std::memcpy(&buf[m], &x_comm_hold[list_i][0], sizeof(double)*x_dim);
      m += x_dim;
    }
  }

  return m;
}

// TODO: refactor for speed
void PairE3GNNParallel::unpack_forward_comm(int n, int first, double *buf) {
  int i,list_i,j,m,last;
  int nlocal = list->inum;
  tagint *tag = atom->tag;
  auto x_ghost_accessor = x_ghost.accessor<float, 2>();

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    /*
       case 1: over pbc, local atom to local atom(same proc) -> just increase 'm'
       case 2: atom received from another process
          case2-1: received atom is known (inside cutoff) -> store to ghost x tensor
          case2-2: received atom in unknown -> store to x_comm_hold and deliver later
       */
    int graph_idx = tag_to_graph_idx_ptr[tag[i]];
    if(graph_idx < nlocal && graph_idx != -1) {
      //case 1, do nothing(local atom x is already complete)
      m += x_dim;
    } else {
      // save to x_comm_hold anyway
      std::memcpy(&x_comm_hold[i][0], &buf[m], sizeof(double)*x_dim);
      m += x_dim;

      if(graph_idx != -1) {
        //case 2-1, given ghost atom is what we wanted
        //save it to x_ghost
        for (j = 0; j < x_dim; j++) {
          x_ghost_accessor[graph_idx-nlocal][j] = static_cast<float>(x_comm_hold[i][j]);
        }
      }
    }
  }
}

int PairE3GNNParallel::pack_reverse_comm(int n, int first, double *buf) {
  buf_hold = buf;

  int i,list_i,j,m,last;
  int nlocal = list->inum;
  tagint *tag = atom->tag;
  auto x_ghost_accessor = x_ghost.accessor<float, 2>();
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    /*
       case 1: comm to self, do nothing. (see unpack reverse)
       case 2-1: comm to other from self (is it really exist?) buf[m] = 0
       case 2-2: comm to other from known ghost atom populate from my ghost atom
       case 2-3: comm to other unknown ghost atom populate from x_comm_hold
       */
    int graph_idx = tag_to_graph_idx_ptr[tag[i]];

    if(graph_idx < nlocal && graph_idx != -1) {
      for (j = 0; j < x_dim; j++) {
        buf[m++] = 0;
      }
    } else if(graph_idx != -1) {
        //case 2-1
        for (j = 0; j < x_dim; j++) {
          buf[m++] = static_cast<double>(x_ghost_accessor[graph_idx-nlocal][j]);
        }
    } else {
        //case 2-2
        std::memcpy(&buf[m], &x_comm_hold[i][0], sizeof(double)*x_dim);
        m += x_dim;
    }
  }
  return m;
}

void PairE3GNNParallel::unpack_reverse_comm(int n, int *list_rev, double *buf) {
  /*
     basically, unpack reverse is "accumulation"
     but in gnn, because it use tag and so duplicated atom exist in input(even ghost)
     In lammps index, there is duplicated atoms and for normal potential these duplicated
     atoms have different forces. but not in GNN, each unique atom has its own unique force
     so using std::set, avoid adding same value twice. this could be accomplished when
     pack_forward_comm by store 0 to buffer or here

     case 1: comm to self. do nothing
     case 2-1: unpack to local atom -> accumulate value
     case 2-2: unpack to known ghost atom -> accumulate value
     case 2-3: unpack to unknown ghost atom -> store to x_comm_hold(should I accumulated?)
     */
  if(buf_hold == buf) {
    // if self comm, do nothing
    return;
  }

  int i,j,m;
  int nlocal = list->inum;
  tagint *tag = atom->tag;
  auto x_local_accessor = x_local.accessor<float, 2>();
  auto x_ghost_accessor = x_ghost.accessor<float, 2>();
  std::set<int> already_met;

  m = 0;
  for (i = 0; i < n; i++) {
    int list_i = list_rev[i];
    int graph_idx = tag_to_graph_idx_ptr[tag[list_i]];

    if(already_met.count(graph_idx) == 1) {
      m += x_dim;
      continue;
    }
    already_met.insert(graph_idx);

    if(graph_idx < nlocal && graph_idx != -1) {
      // if given index is local (case 1) accumulate values to x_local(dE_dx)
      for (j = 0; j < x_dim; j++) {
        x_local_accessor[graph_idx][j] += static_cast<float>(buf[m++]);
      }
    } else if(graph_idx != -1) {
      // to known ghost
      for (j = 0; j < x_dim; j++) {
        x_ghost_accessor[graph_idx-nlocal][j] += static_cast<float>(buf[m++]);
      }
    } else {
      std::memcpy(&x_comm_hold[list_i][0], &buf[m], sizeof(double)*x_dim);
      m += x_dim;
    }
  }
}

