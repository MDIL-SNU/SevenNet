from typing import Final

# see
# https://github.com/pytorch/pytorch/issues/52312
# for FYI

# ~~ keys ~~ #
# PyG : primitive key of torch_geometric.data.Data type

#==================================================#
#~~~~~~~~~~~~~~~~~ KEY for data ~~~~~~~~~~~~~~~~~~ #
#==================================================#
# some raw properties of graph
ATOMIC_NUMBERS: Final[str] = "atomic_numbers"  # (N)
POS: Final[str] = "pos"                        # (N, 3) PyG
CELL: Final[str] = "cell_lattice_vectors"      # (3, 3)
CELL_SHIFT: Final[str] = "pbc_shift"           # (N, 3)
AVG_NUM_NEIGHBOR: Final[str] = "avg_num_neighbor"  # float

EDGE_VEC: Final[str] = "edge_vec"              # (N_edge, 3)
EDGE_LENGTH: Final[str] = "edge_length"        # (N_edge, 1)

# some primary data of graph
EDGE_IDX: Final[str] = "edge_index"            # (2, N_edge) PyG
NODE_FEATURE: Final[str] = "x"                 # (N, ?) PyG
NODE_ATTR: Final[str] = "node_attr"            # (N, N_species) from one_hot
EDGE_ATTR: Final[str] = "edge_attr"            # (from spherical harmonics)
EDGE_EMBEDDING: Final[str] = "edge_embedding"  # (from edge embedding)

# inputs of loss fuction
ENERGY: Final[str] = "total_energy"            # (1)
FORCE: Final[str] = "force_of_atoms"           # (N, 3)
# exist for training
REF_SCALED_PER_ATOM_ENERGY: Final[str] = "reference_scaled_per_atom_energy"
REF_SCALED_FORCE: Final[str] = "reference_scaled_force"

# general outputs of models
ATOMIC_ENERGY: Final[str] = "inferred_atomic_energy"
PRED_TOTAL_ENERGY: Final[str] = "inferred_total_energy"
# This is for training, per atom scale.
###################fix later####################
SCALED_ENERGY: Final[str] = "scaled_total_energy"
SCALED_PER_ATOM_ENERGY: Final[str] = "scaled_per_atom_energy"
###################fix later####################

PRED_FORCE: Final[str] = "inferred_force"
SCALED_FORCE: Final[str] = "scaled_force"

# very general data property for AtomGraphData
NUM_ATOMS: Final[str] = "num_atoms"                # int
NUM_NLOCAL: Final[str] = "nlocal"   # only for lammps parallel
USER_LABEL: Final[str] = "user_label"
BATCH: Final[str] = "batch"

#etc
SELF_CONNECTION_TEMP: Final[str] = "self_cont_tmp"
BATCH_SIZE: Final[str] = "batch_size"

#==================================================#
# ~~~~~~~~ KEY for train configuration ~~~~~~~~~~~ #
#==================================================#
STRUCTURE_LIST: Final[str] = "structure_list"
SAVE_DATASET: Final[str] = "save_dataset_path"
LOAD_DATASET: Final[str] = "load_dataset_path"
FORMAT_OUTPUTS: Final[str] = "format_outputs_for_ase"

RANDOM_SEED: Final[str] = "random_seed"
RATIO: Final[str] = "data_divide_ratio"
USE_TESTSET: Final[str] = "use_testset"
EPOCH: Final[str] = "epoch"
OPTIMIZER: Final[str] = "optimizer"
OPTIM_PARAM: Final[str] = 'optim_param'
SCHEDULER: Final[str] = "scheduler"
SCHEDULER_PARAM: Final[str] = 'scheduler_param'
FORCE_WEIGHT: Final[str] = 'force_loss_weight'
DEVICE: Final[str] = "device"
DTYPE: Final[str] = "dtype"

CONTINUE: Final[str] = "continue"

NUM_WORKERS: Final[str] = "num_workers"  # not work

# ~~~~~~~~ KEY for output configuration ~~~~~~~~~~~ #
SKIP_OUTPUT_UNTIL: Final[str] = "skip_output_until"
DRAW_LC: Final[str] = "draw_learning_curve"

OUTPUT_PER_EPOCH: Final[str] = "output_per_epoch"
# blow 4 keys are child of output_per_epoch
PER_EPOCH: Final[str] = "per_epoch"
DRAW_PARITY: Final[str] = "draw_parity"
MODEL_CHECK_POINT: Final[str] = "model_check_point"
DEPLOY_MODEL: Final[str] = "deploy_model"
SAVE_DATA_PICKLE: Final[str] = "save_data_pickle"

#==================================================#
# ~~~~~~~~ KEY for model configuration ~~~~~~~~~~~ #
#==================================================#
# ~~ global model configuration ~~ #
# note that these names are directly used for input.yaml for user input
MODEL_TYPE: Final[str] = "model_type"
CUTOFF: Final[str] = "cutoff"
CHEMICAL_SPECIES: Final[str] = "chemical_species"
NUM_SPECIES: Final[str] = "number_of_species"
TYPE_MAP: Final[str] = "type_map"

# ~~ E3 equivariant model build configuration keys ~~ #
# see model_build default_config for type
NODE_FEATURE_MULTIPLICITY: Final[str] = "channel"

RADIAL_BASIS: Final[str] = "radial_basis"
BESSEL_BASIS_NUM: Final[str] = "bessel_basis_num"

CUTOFF_FUNCTION: Final[str] = "cutoff_function"
POLY_CUT_P: Final[str] = "poly_cut_p_value"

LMAX: Final[str] = "lmax"
IS_PARITY: Final[str] = "is_parity"
CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS: Final[str] = "weight_nn_hidden_neurons"
NUM_CONVOLUTION: Final[str] = "num_convolution_layer"
ACTIVATION_SCARLAR: Final[str] = "act_scalar"
ACTIVATION_GATE: Final[str] = "act_gate"

RADIAL_BASIS_NAME = "radial_basis_name"
CUTOFF_FUNCTION_NAME = "cutoff_function_name"

AVG_NUM_NEIGHBOR: Final[str] = "avg_num_neigh"
SHIFT: Final[str] = "shift"
SCALE: Final[str] = "scale"
