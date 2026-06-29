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
MODAL_ATTR: Final[str] = (
    'modal_attr'  # (1, N_modalities) for handling multi-modal
)
MODAL_TYPE: Final[str] = 'modal_type'  # (1) one-hot index of modal
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

# temperature related
TEMPERATURE: Final[str] = 'temperature'
TEMPERATURE_ENC = 'temperature_encoding'
TOTAL_ENTROPY: Final[str] = 'total_entropy'
PER_ATOM_ENTROPY: Final[str] = 'per_atom_entropy'
TOTAL_FREE_ENERGY: Final[str] = 'total_free_energy'
PER_ATOM_FREE_ENERGY: Final[str] = 'per_atom_free_energy'
TOTAL_HEAT_CAPACITY: Final[str] = 'total_heat_capacity'
PER_ATOM_HEAT_CAPACITY: Final[str] = 'per_atom_heat_capacity'
TOTAL_ASYMPTOT: Final[str] = 'total_asymptot'
PER_ATOM_ASYMPTOT: Final[str] = 'per_atom_asymptot'

SCALED_ATOMIC_ENTROPY: Final[str] = 'scaled_atomic_entropy'
SCALED_ATOMIC_ASYMPTOT: Final[str] = 'scaled_atomic_asymptot'
ATOMIC_ENTROPY: Final[str] = 'inferred_atomic_entropy'
ATOMIC_ASYMPTOT: Final[str] = 'inferred_atomic_asymptot'
PRED_TOTAL_ENTROPY: Final[str] = 'inferred_total_entropy'
PRED_TOTAL_HEAD_ENTROPY: Final[str] = 'inferred_total_head_entropy'
PRED_TOTAL_HEAD_ASYMPTOT: Final[str] = 'inferred_total_head_asymptot'
PRED_TOTAL_INTERNAL_ENERGY: Final[str] = 'inferred_total_internal_energy'
PRED_TOTAL_FREE_ENERGY: Final[str] = 'inferred_total_free_energy'
PRED_TOTAL_HEAT_CAPACITY: Final[str] = 'inferred_total_heat_capacity'

DEBYE_TEMPERATURE: Final[str] = 'debye_temperature'
DEBYE_FREE_ENERGY: Final[str] = 'debye_free_energy'
DEBYE_INTERNAL_ENERGY: Final[str] = 'debye_internal_energy'
DEBYE_ENTROPY: Final[str] = 'debye_entropy'
DEBYE_HEAT_CAPACITY: Final[str] = 'debye_heat_capacity'
DEBYE_ASYMPTOT: Final[str] = 'debye_asymptot'
DEBYE_ZPE: Final[str] = 'debye_zpe'

# very general data property for AtomGraphData
NUM_ATOMS: Final[str] = 'num_atoms'  # int
NUM_GHOSTS: Final[str] = 'num_ghosts'
NLOCAL: Final[str] = 'nlocal'  # only for lammps parallel, must be on cpu
USER_LABEL: Final[str] = 'user_label'
DATA_WEIGHT: Final[str] = 'data_weight'  # weight for given data
DATA_MODALITY: Final[str] = (
    'data_modality'  # modality of given data. e.g. PBE and SCAN
)
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
LOSS_TYPE = 'loss_type'
LOSS_WEIGHT = 'loss_weight'
OPTIMIZER = 'optimizer'
OPTIM_PARAM = 'optim_param'
SCHEDULER = 'scheduler'
SCHEDULER_PARAM = 'scheduler_param'
SCHEDULER_BATCH_MODE = 'scheduler_batch_mode'
ENERGY_WEIGHT = 'energy_loss_weight'
FORCE_WEIGHT = 'force_loss_weight'
STRESS_WEIGHT = 'stress_loss_weight'
ENTROPY_WEIGHT = 'entropy_loss_weight'
FREE_ENERGY_WEIGHT = 'free_energy_loss_weight'
HEAT_CAPACITY_WEIGHT = 'heat_capacity_loss_weight'
ASYMPTOT_WEIGHT = 'asymptot_loss_weight'
GRAD_CLIP = 'grad_clip'
DEVICE = 'device'
DTYPE = 'dtype'

TRAIN_SHUFFLE = 'train_shuffle'

IS_TRAIN_STRESS = 'is_train_stress'
IS_TRAIN_HEAT_CAPACITY = 'is_train_heat_capacity'
IS_TRAIN_ASYMPTOT = 'is_train_asymptot'

TRAIN_TEMPERATURE_BLOCK_ONLY = 'train_temperature_block_only'
TRAIN_ENERGY_HEAD = 'train_energy_head'
TRAIN_ENTROPY_HEAD = 'train_entropy_head'
TRAIN_ASYMPTOT_HEAD = 'train_asymptot_head'

CONTINUE = 'continue'
CHECKPOINT = 'checkpoint'
RESET_OPTIMIZER = 'reset_optimizer'
RESET_SCHEDULER = 'reset_scheduler'
RESET_EPOCH = 'reset_epoch'
RESET_DATA_PROGRESS = 'reset_data_progress'
USE_STATISTIC_VALUES_OF_CHECKPOINT = 'use_statistic_values_of_checkpoint'
USE_STATISTIC_VALUES_FOR_CP_MODAL_ONLY = (
    'use_statistic_values_for_cp_modal_only'
)

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

TRAIN_BY_BATCH = 'train_by_batch'
TOTAL_DATA_NUM = 'total_data_num'
CURRENT_DATA_IDX = 'current_data_index'
NUMPY_RNG_STATE = 'numpy_rng_state'

USE_WEIGHT = 'use_weight'
USE_MODALITY = 'use_modality'
USE_TEMPERATURE = 'use_temperature'
DEFAULT_MODAL = 'default_modal'


# ==================================================#
# ~~~~~~~~ KEY for model configuration ~~~~~~~~~~~ #
# ==================================================#
# ~~ global model configuration ~~ #
# note that these names are directly used for input.yaml for user input
MODEL_TYPE = '_model_type'
CUTOFF = 'cutoff'
CHEMICAL_SPECIES = 'chemical_species'
MODAL_LIST = 'modal_list'
CHEMICAL_SPECIES_BY_ATOMIC_NUMBER = '_chemical_species_by_atomic_number'
NUM_SPECIES = '_number_of_species'
NUM_MODALITIES = '_number_of_modalities'
TYPE_MAP = '_type_map'
MODAL_MAP = '_modal_map'

# ~~ E3 equivariant model build configuration keys ~~ #
# see model_build default_config for type
IRREPS_MANUAL = 'irreps_manual'
NODE_FEATURE_MULTIPLICITY = 'channel'

RADIAL_BASIS = 'radial_basis'
BESSEL_BASIS_NUM = 'bessel_basis_num'

TEMPERATURE_ENC_FUNC = 'temperature_encoding_function'
TEMPERATURE_ENC_PARAMS = 'temperature_encoding_params'
TEMPERATURE_GATE_FUNCTION = 'temperature_gate_function'
TEMPERATURE_GATE_PARAMS = 'temperature_gate_params'
TEMPERATURE_COEFF_FUNCTION = 'temperature_coeff_function'
TEMPERATURE_COEFF_PARAMS = 'temperature_coeff_params'

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

USE_MODAL_NODE_EMBEDDING = 'use_modal_node_embedding'
USE_MODAL_SELF_INTER_INTRO = 'use_modal_self_inter_intro'
USE_MODAL_SELF_INTER_OUTRO = 'use_modal_self_inter_outro'
USE_MODAL_OUTPUT_BLOCK = 'use_modal_output_block'

USE_TEMPERATURE_NODE_EMBEDDING = 'use_temperature_node_embedding'
USE_TEMPERATURE_SELF_INTER_INTRO = 'use_temperature_self_inter_intro'
USE_TEMPERATURE_SELF_INTER_OUTRO = 'use_temperature_self_inter_outro'
USE_TEMPERATURE_OUTPUT_BLOCK = 'use_temperature_output_block'

READOUT_AS_FCN = 'readout_as_fcn'
READOUT_FCN_HIDDEN_NEURONS = 'readout_fcn_hidden_neurons'
READOUT_FCN_ACTIVATION = 'readout_fcn_activation'

AVG_NUM_NEIGH = 'avg_num_neigh'
CONV_DENOMINATOR = 'conv_denominator'
SHIFT = 'shift'
SCALE = 'scale'
LOADER_KWARGS = 'loader_kwargs'

USE_SPECIES_WISE_SHIFT_SCALE = 'use_species_wise_shift_scale'
USE_MODAL_WISE_SHIFT = 'use_modal_wise_shift'
USE_MODAL_WISE_SCALE = 'use_modal_wise_scale'

TRAIN_SHIFT_SCALE = 'train_shift_scale'
TRAIN_SHIFT = 'train_shift'
TRAIN_SCALE = 'train_scale'
TRAIN_DEBYE_TEMPERATURE = 'train_debye_temperature'
TRAIN_DENOMINTAOR = 'train_denominator'
INTERACTION_TYPE = 'interaction_type'
TRAIN_AVG_NUM_NEIGH = 'train_avg_num_neigh'  # deprecated

USE_FLASH_TP = 'use_flash_tp'
CUEQUIVARIANCE_CONFIG = 'cuequivariance_config'
USE_OEQ = 'use_oeq'

REG_PARAM = 'regularization_param'
REG_WEIGHT = 'regularization_weight'

_NORMALIZE_SPH = '_normalize_sph'
OPTIMIZE_BY_REDUCE = 'optimize_by_reduce'

# MLIAP_related
USE_MLIAP = 'use_mliap'
MLIAP_NUM_LOCAL_GHOST = 'mliap_num_local_ghost'
MLIAP_NODE_FEATURE_GHOST = 'mliap_node_feature_ghost'
LAMMPS_DATA = 'lammps_data'
