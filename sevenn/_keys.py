"""
How to add new feature?

1. Add new key to this file.
2. Add new key to _const.py
2.1. if the type of input is consistent,
    write adequate condition and default to _const.py.
2.2. if the type of input is not consistent,
    you must add your own input validation code to
    parse_input.py
"""

from typing import Final

from torch.jit import CompilationUnit

# see
# https://github.com/pytorch/pytorch/issues/52312
# for FYI

# ~~ keys ~~ #
# PyG : primitive key of torch_geometric.data.Data type

# ==================================================#
# ~~~~~~~~~~~~~~~~~ KEY for data ~~~~~~~~~~~~~~~~~~ #
# ==================================================#
# some raw properties of graph
ATOMIC_NUMBERS: Final[str] = 'atomic_numbers'  # (N)
POS: Final[str] = 'pos'  # (N, 3) PyG
CELL: Final[str] = 'cell_lattice_vectors'  # (3, 3)
CELL_SHIFT: Final[str] = 'pbc_shift'  # (N, 3)
CELL_VOLUME: Final[str] = 'cell_volume'

EDGE_VEC: Final[str] = 'edge_vec'  # (N_edge, 3)
EDGE_LENGTH: Final[str] = 'edge_length'  # (N_edge, 1)

# some primary data of graph
EDGE_IDX: Final[str] = 'edge_index'  # (2, N_edge) PyG
ATOM_TYPE: Final[str] = 'atom_type'  # (N) one-hot index of nodes
NODE_FEATURE: Final[str] = 'x'  # (N, ?) PyG
NODE_FEATURE_GHOST: Final[str] = 'x_ghost'
NODE_ATTR: Final[str] = 'node_attr'  # (N, N_species) from one_hot
EDGE_ATTR: Final[str] = 'edge_attr'  # (from spherical harmonics)
EDGE_EMBEDDING: Final[str] = 'edge_embedding'  # (from edge embedding)

# inputs of loss function
ENERGY: Final[str] = 'total_energy'  # (1)
FORCE: Final[str] = 'force_of_atoms'  # (N, 3)
STRESS: Final[str] = 'stress'  # (6)

# This is for training, per atom scale.
SCALED_ENERGY: Final[str] = 'scaled_total_energy'

# general outputs of models
SCALED_ATOMIC_ENERGY: Final[str] = 'scaled_atomic_energy'
ATOMIC_ENERGY: Final[str] = 'atomic_energy'
PRED_TOTAL_ENERGY: Final[str] = 'inferred_total_energy'

PRED_PER_ATOM_ENERGY: Final[str] = 'inferred_per_atom_energy'
PER_ATOM_ENERGY: Final[str] = 'per_atom_energy'

PRED_FORCE: Final[str] = 'inferred_force'
SCALED_FORCE: Final[str] = 'scaled_force'

PRED_STRESS: Final[str] = 'inferred_stress'
SCALED_STRESS: Final[str] = 'scaled_stress'

# very general data property for AtomGraphData
NUM_ATOMS: Final[str] = 'num_atoms'  # int
NUM_GHOSTS: Final[str] = 'num_ghosts'
NLOCAL: Final[str] = 'nlocal'  # only for lammps parallel, must be on cpu
USER_LABEL: Final[str] = 'user_label'  # Deprecated from v0.9.6
BATCH: Final[str] = 'batch'

TAG = 'tag'  # replace USER_LABEL

# etc
SELF_CONNECTION_TEMP: Final[str] = 'self_cont_tmp'
BATCH_SIZE: Final[str] = 'batch_size'
INFO: Final[str] = 'data_info'

# something special
LABEL_NONE: Final[str] = 'No_label'

# ==================================================#
# ~~~~~~ KEY for train/data configuration ~~~~~~~~ #
# ==================================================#
PREPROCESS_NUM_CORES = 'preprocess_num_cores'
SAVE_DATASET = 'save_dataset_path'
SAVE_BY_LABEL = 'save_by_label'
SAVE_BY_TRAIN_VALID = 'save_by_train_valid'
DATA_FORMAT = 'data_format'
DATA_FORMAT_ARGS = 'data_format_args'
STRUCTURE_LIST = 'structure_list'
LOAD_DATASET = 'load_dataset_path'  # not used in v2
LOAD_TRAINSET = 'load_trainset_path'
LOAD_VALIDSET = 'load_validset_path'
LOAD_TESTSET = 'load_testset_path'
FORMAT_OUTPUTS = 'format_outputs_for_ase'
COMPUTE_STATISTICS = 'compute_statistics'
DATASET_TYPE = 'dataset_type'

RANDOM_SEED = 'random_seed'
RATIO = 'data_divide_ratio'
USE_TESTSET = 'use_testset'
EPOCH = 'epoch'
LOSS = 'loss'
LOSS_PARAM = 'loss_param'
OPTIMIZER = 'optimizer'
OPTIM_PARAM = 'optim_param'
SCHEDULER = 'scheduler'
SCHEDULER_PARAM = 'scheduler_param'
FORCE_WEIGHT = 'force_loss_weight'
STRESS_WEIGHT = 'stress_loss_weight'
DEVICE = 'device'
DTYPE = 'dtype'

TRAIN_SHUFFLE = 'train_shuffle'

IS_TRAIN_STRESS = 'is_train_stress'

CONTINUE = 'continue'
CHECKPOINT = 'checkpoint'
RESET_OPTIMIZER = 'reset_optimizer'
RESET_SCHEDULER = 'reset_scheduler'
RESET_EPOCH = 'reset_epoch'
USE_STATISTIC_VALUES_OF_CHECKPOINT = 'use_statistic_values_of_checkpoint'

CSV_LOG = 'csv_log'

ERROR_RECORD = 'error_record'
BEST_METRIC = 'best_metric'

NUM_WORKERS = 'num_workers'  # not work

RANK = 'rank'
LOCAL_RANK = 'local_rank'
WORLD_SIZE = 'world_size'
IS_DDP = 'is_ddp'
DDP_BACKEND = 'ddp_backend'
PER_EPOCH = 'per_epoch'


# ==================================================#
# ~~~~~~~~ KEY for model configuration ~~~~~~~~~~~ #
# ==================================================#
# ~~ global model configuration ~~ #
# note that these names are directly used for input.yaml for user input
MODEL_TYPE = '_model_type'
CUTOFF = 'cutoff'
CHEMICAL_SPECIES = 'chemical_species'
CHEMICAL_SPECIES_BY_ATOMIC_NUMBER = '_chemical_species_by_atomic_number'
NUM_SPECIES = '_number_of_species'
TYPE_MAP = '_type_map'

# ~~ E3 equivariant model build configuration keys ~~ #
# see model_build default_config for type
IRREPS_MANUAL = 'irreps_manual'
NODE_FEATURE_MULTIPLICITY = 'channel'

RADIAL_BASIS = 'radial_basis'
BESSEL_BASIS_NUM = 'bessel_basis_num'

CUTOFF_FUNCTION = 'cutoff_function'
POLY_CUT_P = 'poly_cut_p_value'

LMAX = 'lmax'
LMAX_EDGE = 'lmax_edge'
LMAX_NODE = 'lmax_node'
IS_PARITY = 'is_parity'
CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS = 'weight_nn_hidden_neurons'
NUM_CONVOLUTION = 'num_convolution_layer'
ACTIVATION_SCARLAR = 'act_scalar'
ACTIVATION_GATE = 'act_gate'
ACTIVATION_RADIAL = 'act_radial'

SELF_CONNECTION_TYPE = 'self_connection_type'

RADIAL_BASIS_NAME = 'radial_basis_name'
CUTOFF_FUNCTION_NAME = 'cutoff_function_name'

USE_BIAS_IN_LINEAR = 'use_bias_in_linear'

READOUT_AS_FCN = 'readout_as_fcn'
READOUT_FCN_HIDDEN_NEURONS = 'readout_fcn_hidden_neurons'
READOUT_FCN_ACTIVATION = 'readout_fcn_activation'

AVG_NUM_NEIGH = 'avg_num_neigh'
CONV_DENOMINATOR = 'conv_denominator'
SHIFT = 'shift'
SCALE = 'scale'

USE_SPECIES_WISE_SHIFT_SCALE = 'use_species_wise_shift_scale'

TRAIN_SHIFT_SCALE = 'train_shift_scale'
TRAIN_DENOMINTAOR = 'train_denominator'
INTERACTION_TYPE = 'interaction_type'
TRAIN_AVG_NUM_NEIGH = 'train_avg_num_neigh'  # deprecated

_NORMALIZE_SPH = '_normalize_sph'
OPTIMIZE_BY_REDUCE = 'optimize_by_reduce'
