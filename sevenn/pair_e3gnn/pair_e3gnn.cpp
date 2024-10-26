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

#include <torch/script.h>
#include <torch/torch.h>

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

#include "pair_e3gnn.h"

using namespace LAMMPS_NS;

#define INTEGER_TYPE torch::TensorOptions().dtype(torch::kInt64)
#define FLOAT_TYPE torch::TensorOptions().dtype(torch::kFloat)

PairE3GNN::PairE3GNN(LAMMPS *lmp) : Pair(lmp) {
  // constructor
  const char *print_flag = std::getenv("SEVENN_PRINT_INFO");
  if (print_flag)
    print_info = true;

  std::string device_name;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
    device_name = "CUDA";
  } else {
    device = torch::kCPU;
    device_name = "CPU";
  }

  if (lmp->logfile) {
    fprintf(lmp->logfile, "PairE3GNN using device : %s\n", device_name.c_str());
  }
}

PairE3GNN::~PairE3GNN() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(map);
    memory->destroy(elements);
  }
}

void PairE3GNN::compute(int eflag, int vflag) {
  // compute
  /*
     This compute function is ispired/modified from stress branch of pair-nequip
     https://github.com/mir-group/pair_nequip
  */

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;
  if (vflag_atom) {
    error->all(FLERR, "atomic stress is not supported\n");
  }

  int nlocal = list->inum; // same as nlocal
  int *ilist = list->ilist;
  tagint *tag = atom->tag;
  std::unordered_map<int, int> tag_map;

  if (atom->tag_consecutive() == 0) {
    for (int ii = 0; ii < nlocal; ii++) {
      const int i = ilist[ii];
      int itag = tag[i];
      tag_map[itag] = ii+1;
      // printf("MODIFY setting %i => %i \n",itag, tag_map[itag] );
    }
  } else {
    //Ordered which mappling required
    for (int ii = 0; ii < nlocal; ii++) {
        const int itag = ilist[ii]+1;
        tag_map[itag] = ii+1;
        // printf("normal setting %i => %i \n",itag, tag_map[itag] );
    }
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  long num_atoms[1] = {nlocal};

  int tag2i[nlocal];

  int *numneigh = list->numneigh;      // j loop cond
  int **firstneigh = list->firstneigh; // j list

  int bound;
  if (this->nedges_bound == -1) {
    bound = std::accumulate(numneigh, numneigh + nlocal, 0);
  } else {
    bound = this->nedges_bound;
  }
  const int nedges_upper_bound = bound;

  float cell[3][3];
  cell[0][0] = domain->boxhi[0] - domain->boxlo[0];
  cell[0][1] = 0.0;
  cell[0][2] = 0.0;

  cell[1][0] = domain->xy;
  cell[1][1] = domain->boxhi[1] - domain->boxlo[1];
  cell[1][2] = 0.0;

  cell[2][0] = domain->xz;
  cell[2][1] = domain->yz;
  cell[2][2] = domain->boxhi[2] - domain->boxlo[2];

  torch::Tensor inp_cell = torch::from_blob(cell, {3, 3}, FLOAT_TYPE);
  torch::Tensor inp_num_atoms = torch::from_blob(num_atoms, {1}, INTEGER_TYPE);

  torch::Tensor inp_node_type = torch::zeros({nlocal}, INTEGER_TYPE);
  torch::Tensor inp_pos = torch::zeros({nlocal, 3});

  torch::Tensor inp_cell_volume =
      torch::dot(inp_cell[0], torch::cross(inp_cell[1], inp_cell[2], 0));

  float pbc_shift_tmp[nedges_upper_bound][3];

  auto node_type = inp_node_type.accessor<long, 1>();
  auto pos = inp_pos.accessor<float, 2>();

  long edge_idx_src[nedges_upper_bound];
  long edge_idx_dst[nedges_upper_bound];

  int nedges = 0;

  for (int ii = 0; ii < nlocal; ii++) {
    const int i = ilist[ii];
    int itag = tag_map[tag[i]];
    tag2i[itag - 1] = i;
    const int itype = type[i];
    node_type[itag - 1] = map[itype];
    pos[itag - 1][0] = x[i][0];
    pos[itag - 1][1] = x[i][1];
    pos[itag - 1][2] = x[i][2];
  }

  for (int ii = 0; ii < nlocal; ii++) {
    const int i = ilist[ii];
    int itag = tag_map[tag[i]];
    const int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj]; // atom over pbc is different atom
      int jtag = tag_map[tag[j]]; // atom over pbs is same atom (it starts from 1)
      j &= NEIGHMASK;
      const int jtype = type[j];

      const double delij[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1],
                               x[j][2] - x[i][2]};
      const double Rij =
          delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
      if (Rij < cutoff_square) {
        edge_idx_src[nedges] = itag - 1;
        edge_idx_dst[nedges] = jtag - 1;

        pbc_shift_tmp[nedges][0] = x[j][0] - pos[jtag - 1][0];
        pbc_shift_tmp[nedges][1] = x[j][1] - pos[jtag - 1][1];
        pbc_shift_tmp[nedges][2] = x[j][2] - pos[jtag - 1][2];

        nedges++;
      }
    } // j loop end
  }   // i loop end

  auto edge_idx_src_tensor =
      torch::from_blob(edge_idx_src, {nedges}, INTEGER_TYPE);
  auto edge_idx_dst_tensor =
      torch::from_blob(edge_idx_dst, {nedges}, INTEGER_TYPE);
  auto inp_edge_index =
      torch::stack({edge_idx_src_tensor, edge_idx_dst_tensor});

  // r' = r + {shift_tensor(integer vector of len 3)} @ cell_tensor
  // shift_tensor = (cell_tensor)^-1^T @ (r' - r)
  torch::Tensor cell_inv_tensor =
      inp_cell.inverse().transpose(0, 1).unsqueeze(0).to(device);
  torch::Tensor pbc_shift_tmp_tensor =
      torch::from_blob(pbc_shift_tmp, {nedges, 3}, FLOAT_TYPE)
          .view({nedges, 3, 1})
          .to(device);
  torch::Tensor inp_cell_shift =
      torch::bmm(cell_inv_tensor.expand({nedges, 3, 3}), pbc_shift_tmp_tensor)
          .view({nedges, 3});

  inp_pos.set_requires_grad(true);

  c10::Dict<std::string, torch::Tensor> input_dict;
  input_dict.insert("x", inp_node_type.to(device));
  input_dict.insert("pos", inp_pos.to(device));
  input_dict.insert("edge_index", inp_edge_index.to(device));
  input_dict.insert("num_atoms", inp_num_atoms.to(device));
  input_dict.insert("cell_lattice_vectors", inp_cell.to(device));
  input_dict.insert("cell_volume", inp_cell_volume.to(device));
  input_dict.insert("pbc_shift", inp_cell_shift);

  std::vector<torch::IValue> input(1, input_dict);
  auto output = model.forward(input).toGenericDict();

  torch::Tensor total_energy_tensor =
      output.at("inferred_total_energy").toTensor().cpu();
  torch::Tensor force_tensor = output.at("inferred_force").toTensor().cpu();
  auto forces = force_tensor.accessor<float, 2>();
  eng_vdwl += total_energy_tensor.item<float>();

  for (int itag = 0; itag < nlocal; itag++) {
    int i = tag2i[itag];
    f[i][0] += forces[itag][0];
    f[i][1] += forces[itag][1];
    f[i][2] += forces[itag][2];
  }

  if (vflag) {
    // more accurately, it is virial part of stress
    torch::Tensor stress_tensor = output.at("inferred_stress").toTensor().cpu();
    auto virial_stress_tensor = stress_tensor * inp_cell_volume;
    // xy yz zx order in vasp (voigt is xx yy zz yz xz xy)
    auto virial_stress = virial_stress_tensor.accessor<float, 1>();
    virial[0] += virial_stress[0];
    virial[1] += virial_stress[1];
    virial[2] += virial_stress[2];
    virial[3] += virial_stress[3];
    virial[4] += virial_stress[5];
    virial[5] += virial_stress[4];
  }

  if (eflag_atom) {
    torch::Tensor atomic_energy_tensor =
        output.at("atomic_energy").toTensor().cpu().squeeze();
    auto atomic_energy = atomic_energy_tensor.accessor<float, 1>();
    for (int itag = 0; itag < nlocal; itag++) {
      int i = tag2i[itag];
      eatom[i] += atomic_energy[itag];
    }
  }

  // if it was the first MD step
  if (this->nedges_bound == -1) {
    this->nedges_bound = nedges * 1.2;
  } // else if the nedges is too small, increase the bound
  else if (nedges > this->nedges_bound / 1.2) {
    this->nedges_bound = nedges * 1.2;
  }
}

// allocate arrays (called from coeff)
void PairE3GNN::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(map, n + 1, "pair:map");
}

// global settings for pair_style
void PairE3GNN::settings(int narg, char **arg) {
  if (narg != 0) {
    error->all(FLERR, "Illegal pair_style command");
  }
}

void PairE3GNN::coeff(int narg, char **arg) {

  if (allocated) {
    error->all(FLERR, "pair_e3gnn coeff called twice");
  }
  allocate();

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0) {
    error->all(FLERR,
               "e3gnn: first and second input of pair_coeff should be '*'");
  }
  // expected input : pair_coeff * * pot.pth type_name1 type_name2 ...

  std::unordered_map<std::string, std::string> meta_dict = {
      {"chemical_symbols_to_index", ""},
      {"cutoff", ""},
      {"num_species", ""},
      {"model_type", ""},
      {"version", ""},
      {"dtype", ""},
      {"time", ""}};

  // model loading from input
  try {
    model = torch::jit::load(std::string(arg[2]), device, meta_dict);
  } catch (const c10::Error &e) {
    error->all(FLERR, "error loading the model, check the path of the model");
  }
  // model = torch::jit::freeze(model); model is already freezed

  torch::jit::setGraphExecutorOptimize(false);
  torch::jit::FusionStrategy strategy;
  // thing about dynamic recompile as tensor shape varies, this is default
  // strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  strategy = {{torch::jit::FusionBehavior::STATIC, 0}};
  torch::jit::setFusionStrategy(strategy);

  cutoff = std::stod(meta_dict["cutoff"]);
  cutoff_square = cutoff * cutoff;

  if (meta_dict["model_type"].compare("E3_equivariant_model") != 0) {
    error->all(FLERR, "given model type is not E3_equivariant_model");
  }

  std::string chem_str = meta_dict["chemical_symbols_to_index"];
  int ntypes = atom->ntypes;

  auto delim = " ";
  char *tok = std::strtok(const_cast<char *>(chem_str.c_str()), delim);
  std::vector<std::string> chem_vec;
  while (tok != nullptr) {
    chem_vec.push_back(std::string(tok));
    tok = std::strtok(nullptr, delim);
  }

  bool found_flag = false;
  for (int i = 3; i < narg; i++) {
    found_flag = false;
    for (int j = 0; j < chem_vec.size(); j++) {
      if (chem_vec[j].compare(arg[i]) == 0) {
        map[i - 2] = j;
        found_flag = true;
        fprintf(lmp->logfile, "Chemical specie '%s' is assigned to type %d\n",
                arg[i], i - 2);
        break;
      }
    }
    if (!found_flag) {
      error->all(FLERR, "Unknown chemical specie is given");
    }
  }

  if (ntypes > narg - 3) {
    error->all(FLERR, "Not enough chemical specie is given. Check pair_coeff "
                      "and types in your data/script");
  }

  for (int i = 1; i <= ntypes; i++) {
    for (int j = 1; j <= ntypes; j++) {
      if ((map[i] >= 0) && (map[j] >= 0)) {
        setflag[i][j] = 1;
        cutsq[i][j] = cutoff * cutoff;
      }
    }
  }

  if (lmp->logfile) {
    fprintf(lmp->logfile, "from sevenn version '%s' ",
            meta_dict["version"].c_str());
    fprintf(lmp->logfile, "%s precision model trained at %s is loaded\n",
            meta_dict["dtype"].c_str(), meta_dict["time"].c_str());
  }
}

// init specific to this pair
void PairE3GNN::init_style() {
  // Newton flag is irrelevant if use only one processor for simulation
  /*
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style nn requires newton pair on");
  }
  */

  // full neighbor list (this is many-body potential)
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairE3GNN::init_one(int i, int j) { return cutoff; }
