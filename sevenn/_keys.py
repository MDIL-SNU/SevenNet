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
CELL_VOLUME: Final[str] = "cell_volume"
AVG_NUM_NEIGHBOR: Final[str] = "avg_num_neighbor"  # float

EDGE_VEC: Final[str] = "edge_vec"              # (N_edge, 3)
EDGE_LENGTH: Final[str] = "edge_length"        # (N_edge, 1)

# some primary data of graph
EDGE_IDX: Final[str] = "edge_index"            # (2, N_edge) PyG
NODE_FEATURE: Final[str] = "x"                 # (N, ?) PyG
NODE_FEATURE_GHOST: Final[str] = "x_ghost"
NODE_ATTR: Final[str] = "node_attr"            # (N, N_species) from one_hot
EDGE_ATTR: Final[str] = "edge_attr"            # (from spherical harmonics)
EDGE_EMBEDDING: Final[str] = "edge_embedding"  # (from edge embedding)

# inputs of loss fuction
ENERGY: Final[str] = "total_energy"            # (1)
FORCE: Final[str] = "force_of_atoms"           # (N, 3)
STRESS: Final[str] = "stress"                  # (6)

# This is for training, per atom scale.
SCALED_ENERGY: Final[str] = "scaled_total_energy"

# general outputs of models
SCALED_ATOMIC_ENERGY: Final[str] = "scaled_atomic_energy"
PRED_TOTAL_ENERGY: Final[str] = "inferred_total_energy"

#TODO: remove later (after doing moving nn.Rescale TODO)
PRED_PER_ATOM_ENERGY: Final[str] = "inferred_per_atom_energy"
PER_ATOM_ENERGY: Final[str] = "per_atom_energy"

PRED_FORCE: Final[str] = "inferred_force"
SCALED_FORCE: Final[str] = "scaled_force"

PRED_STRESS: Final[str] = "inferred_stress"
SCALED_STRESS: Final[str] = "scaled_stress"

# very general data property for AtomGraphData
NUM_ATOMS: Final[str] = "num_atoms"                # int
NUM_GHOSTS: Final[str] = "num_ghosts"
NLOCAL: Final[str] = "nlocal"   # only for lammps parallel, must be on cpu
USER_LABEL: Final[str] = "user_label"
BATCH: Final[str] = "batch"

#etc
SELF_CONNECTION_TEMP: Final[str] = "self_cont_tmp"
BATCH_SIZE: Final[str] = "batch_size"
INFO: Final[str] = "data_info"


#==================================================#
# ~~~~~~~~ KEY for train configuration ~~~~~~~~~~~ #
#==================================================#
PREPROCESS_NUM_CORES: Final[str] = "preprocess_num_cores"
SAVE_DATASET: Final[str] = "save_dataset_path"
SAVE_BY_LABEL: Final[str] = "save_by_label"
SAVE_BY_TRAIN_VALID: Final[str] = "save_by_train_valid"
STRUCTURE_LIST: Final[str] = "structure_list"
LOAD_DATASET: Final[str] = "load_dataset_path"
LOAD_VALIDSET: Final[str] = "load_validset_path"
FORMAT_OUTPUTS: Final[str] = "format_outputs_for_ase"

RANDOM_SEED: Final[str] = "random_seed"
RATIO: Final[str] = "data_divide_ratio"
USE_TESTSET: Final[str] = "use_testset"
EPOCH: Final[str] = "epoch"
LOSS: Final[str] = "loss"
LOSS_PARAM: Final[str] = "loss_param"
OPTIMIZER: Final[str] = "optimizer"
OPTIM_PARAM: Final[str] = 'optim_param'
SCHEDULER: Final[str] = "scheduler"
SCHEDULER_PARAM: Final[str] = 'scheduler_param'
FORCE_WEIGHT: Final[str] = 'force_loss_weight'
STRESS_WEIGHT: Final[str] = 'stress_loss_weight'
DEVICE: Final[str] = "device"
DTYPE: Final[str] = "dtype"

IS_TRACE_STRESS: Final[str] = "_is_trace_stress"
IS_TRAIN_STRESS: Final[str] = "is_train_stress"

CONTINUE: Final[str] = "continue"
CHECKPOINT: Final[str] = "checkpoint"
RESET_OPTIMIZER: Final[str] = "reset_optimizer"
RESET_SCHEDULER: Final[str] = "reset_scheduler"

NUM_WORKERS: Final[str] = "_num_workers"  # not work

RANK: Final[str] = "rank"
LOCAL_RANK: Final[str] = "local_rank"
WORLD_SIZE: Final[str] = "world_size"
IS_DDP : Final[str] = "is_ddp"
# ~~~~~~~~ KEY for output configuration ~~~~~~~~~~~ #

# blow 4 keys are child of output_per_epoch
PER_EPOCH: Final[str] = "per_epoch"


#==================================================#
# ~~~~~~~~ KEY for model configuration ~~~~~~~~~~~ #
#==================================================#
# ~~ global model configuration ~~ #
# note that these names are directly used for input.yaml for user input
MODEL_TYPE: Final[str] = "_model_type"
CUTOFF: Final[str] = "cutoff"
CHEMICAL_SPECIES: Final[str] = "chemical_species"
CHEMICAL_SPECIES_BY_ATOMIC_NUMBER: Final[str] = "_chemical_species_by_atomic_number"
NUM_SPECIES: Final[str] = "_number_of_species"
TYPE_MAP: Final[str] = "_type_map"

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

USE_BIAS_IN_LINEAR = "use_bias_in_linear"

AVG_NUM_NEIGHBOR: Final[str] = "avg_num_neigh"
SHIFT: Final[str] = "shift"
SCALE: Final[str] = "scale"
TRAIN_SHIFT_SCALE: Final[str] = "train_shift_scale"
TRAIN_AVG_NUM_NEIGH: Final[str] = "train_avg_num_neigh"

OPTIMIZE_BY_REDUCE: Final[str] = "optimize_by_reduce"

# deprecated
DRAW_PARITY: Final[str] = "draw_parity"
MODEL_CHECK_POINT: Final[str] = "model_check_point"
DEPLOY_MODEL: Final[str] = "deploy_model"
SAVE_DATA_PICKLE: Final[str] = "save_data_pickle"
SKIP_OUTPUT_UNTIL: Final[str] = "skip_output_until"
DRAW_LC: Final[str] = "draw_learning_curve"
OUTPUT_PER_EPOCH: Final[str] = "output_per_epoch"
