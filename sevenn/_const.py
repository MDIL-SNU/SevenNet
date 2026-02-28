import os
from enum import Enum
from typing import Dict

import torch

import sevenn._keys as KEY
from sevenn.nn.activation import ShiftedSoftPlus

NUM_UNIV_ELEMENT = 119  # Z = 0 ~ 118

IMPLEMENTED_RADIAL_BASIS = ['bessel']
IMPLEMENTED_CUTOFF_FUNCTION = ['poly_cut', 'XPLOR']
# TODO: support None. This became difficult because of parallel model
IMPLEMENTED_SELF_CONNECTION_TYPE = ['nequip', 'linear']
IMPLEMENTED_INTERACTION_TYPE = ['nequip']

IMPLEMENTED_SHIFT = ['per_atom_energy_mean', 'elemwise_reference_energies']
IMPLEMENTED_SCALE = ['force_rms', 'per_atom_energy_std', 'elemwise_force_rms']

SUPPORTING_METRICS = ['RMSE', 'ComponentRMSE', 'MAE', 'Loss']
SUPPORTING_ERROR_TYPES = [
    'TotalEnergy',
    'Energy',
    'Force',
    'Stress',
    'Stress_GPa',
    'TotalLoss',
]

IMPLEMENTED_MODEL = ['E3_equivariant_model']

# string input to real torch function
ACTIVATION = {
    'relu': torch.nn.functional.relu,
    'silu': torch.nn.functional.silu,
    'tanh': torch.tanh,
    'abs': torch.abs,
    'ssp': ShiftedSoftPlus,
    'sigmoid': torch.sigmoid,
    'elu': torch.nn.functional.elu,
}
ACTIVATION_FOR_EVEN = {
    'ssp': ShiftedSoftPlus,
    'silu': torch.nn.functional.silu,
}
ACTIVATION_FOR_ODD = {'tanh': torch.tanh, 'abs': torch.abs}
ACTIVATION_DICT = {'e': ACTIVATION_FOR_EVEN, 'o': ACTIVATION_FOR_ODD}

_prefix = os.path.abspath(f'{os.path.dirname(__file__)}/pretrained_potentials')
SEVENNET_0_11Jul2024 = f'{_prefix}/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth'
SEVENNET_0_22May2024 = f'{_prefix}/SevenNet_0__22May2024/checkpoint_sevennet_0.pth'
SEVENNET_l3i5 = f'{_prefix}/SevenNet_l3i5/checkpoint_l3i5.pth'
SEVENNET_MF_0 = f'{_prefix}/SevenNet_MF_0/checkpoint_sevennet_mf_0.pth'
SEVENNET_MF_ompa = f'{_prefix}/SevenNet_MF_ompa/checkpoint_sevennet_mf_ompa.pth'
SEVENNET_omat = f'{_prefix}/SevenNet_omat/checkpoint_sevennet_omat.pth'
SEVENNET_omni = f'{_prefix}/SevenNet_omni/checkpoint_sevennet_omni.pth'
SEVENNET_omni_i8 = f'{_prefix}/SevenNet_omni_i8/checkpoint_sevennet_omni_i8.pth'
SEVENNET_omni_i12 = f'{_prefix}/SevenNet_omni_i12/checkpoint_sevennet_omni_i12.pth'

_git_prefix = 'https://github.com/MDIL-SNU/SevenNet/releases/download'
CHECKPOINT_DOWNLOAD_LINKS = {
    SEVENNET_MF_ompa: f'{_git_prefix}/v0.11.0.cp/checkpoint_sevennet_mf_ompa.pth',
    SEVENNET_omat: f'{_git_prefix}/v0.11.0.cp/checkpoint_sevennet_omat.pth',
    SEVENNET_omni: f'{_git_prefix}/v0.12.0.cp/checkpoint_sevennet_omni.pth',
    SEVENNET_omni_i8: f'{_git_prefix}/v0.12.1.cp/checkpoint_sevennet_omni_i8.pth',
    SEVENNET_omni_i12: f'{_git_prefix}/v0.12.1.cp/checkpoint_sevennet_omni_i12.pth',
}
# to avoid torch script to compile torch_geometry.data
AtomGraphDataType = Dict[str, torch.Tensor]  # But it can contain 'lmp_data'
# AtomGraphDataType = Dict[str, any]


class LossType(Enum):  # only used for train_v1, do not use it afterwards
    ENERGY = 'energy'  # eV or eV/atom
    FORCE = 'force'  # eV/A
    STRESS = 'stress'  # kB


def error_record_condition(x):
    if type(x) is not list:
        return False
    for v in x:
        if type(v) is not list or len(v) != 2:
            return False
        if v[0] not in SUPPORTING_ERROR_TYPES:
            return False
        if v[0] == 'TotalLoss':
            continue
        if v[1] not in SUPPORTING_METRICS:
            return False
    return True


DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG = {
    KEY.CUTOFF: 4.5,
    KEY.NODE_FEATURE_MULTIPLICITY: 32,
    KEY.IRREPS_MANUAL: False,
    KEY.LMAX: 1,
    KEY.LMAX_EDGE: -1,  # -1 means lmax_edge = lmax
    KEY.LMAX_NODE: -1,  # -1 means lmax_node = lmax
    KEY.IS_PARITY: True,
    KEY.NUM_CONVOLUTION: 3,
    KEY.RADIAL_BASIS: {
        KEY.RADIAL_BASIS_NAME: 'bessel',
    },
    KEY.CUTOFF_FUNCTION: {
        KEY.CUTOFF_FUNCTION_NAME: 'poly_cut',
    },
    KEY.ACTIVATION_RADIAL: 'silu',
    KEY.ACTIVATION_SCARLAR: {'e': 'silu', 'o': 'tanh'},
    KEY.ACTIVATION_GATE: {'e': 'silu', 'o': 'tanh'},
    KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS: [64, 64],
    # KEY.AVG_NUM_NEIGH: True,  # deprecated
    # KEY.TRAIN_AVG_NUM_NEIGH: False,  # deprecated
    KEY.CONV_DENOMINATOR: 'avg_num_neigh',
    KEY.TRAIN_DENOMINTAOR: False,
    KEY.TRAIN_SHIFT_SCALE: False,
    # KEY.OPTIMIZE_BY_REDUCE: True,  # deprecated, always True
    KEY.USE_BIAS_IN_LINEAR: False,
    KEY.USE_MODAL_NODE_EMBEDDING: False,
    KEY.USE_MODAL_SELF_INTER_INTRO: False,
    KEY.USE_MODAL_SELF_INTER_OUTRO: False,
    KEY.USE_MODAL_OUTPUT_BLOCK: False,
    KEY.READOUT_AS_FCN: False,
    # Applied af readout as fcn is True
    KEY.READOUT_FCN_HIDDEN_NEURONS: [30, 30],
    KEY.READOUT_FCN_ACTIVATION: 'relu',
    KEY.SELF_CONNECTION_TYPE: 'nequip',
    KEY.INTERACTION_TYPE: 'nequip',
    KEY._NORMALIZE_SPH: True,
    KEY.USE_FLASH_TP: False,
    KEY.CUEQUIVARIANCE_CONFIG: {},
    KEY.USE_OEQ: False,
}


# Basically, "If provided, it should be type of ..."
MODEL_CONFIG_CONDITION = {
    KEY.NODE_FEATURE_MULTIPLICITY: int,
    KEY.LMAX: int,
    KEY.LMAX_EDGE: int,
    KEY.LMAX_NODE: int,
    KEY.IS_PARITY: bool,
    KEY.RADIAL_BASIS: {
        KEY.RADIAL_BASIS_NAME: lambda x: x in IMPLEMENTED_RADIAL_BASIS,
    },
    KEY.CUTOFF_FUNCTION: {
        KEY.CUTOFF_FUNCTION_NAME: lambda x: x in IMPLEMENTED_CUTOFF_FUNCTION,
    },
    KEY.CUTOFF: float,
    KEY.NUM_CONVOLUTION: int,
    KEY.CONV_DENOMINATOR: lambda x: isinstance(x, float)
    or x
    in [
        'avg_num_neigh',
        'sqrt_avg_num_neigh',
    ],
    KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS: list,
    KEY.TRAIN_SHIFT_SCALE: bool,
    KEY.TRAIN_DENOMINTAOR: bool,
    KEY.USE_BIAS_IN_LINEAR: bool,
    KEY.USE_MODAL_NODE_EMBEDDING: bool,
    KEY.USE_MODAL_SELF_INTER_INTRO: bool,
    KEY.USE_MODAL_SELF_INTER_OUTRO: bool,
    KEY.USE_MODAL_OUTPUT_BLOCK: bool,
    KEY.READOUT_AS_FCN: bool,
    KEY.READOUT_FCN_HIDDEN_NEURONS: list,
    KEY.READOUT_FCN_ACTIVATION: str,
    KEY.ACTIVATION_RADIAL: str,
    KEY.SELF_CONNECTION_TYPE: lambda x: (
        x in IMPLEMENTED_SELF_CONNECTION_TYPE
        or (
            isinstance(x, list)
            and all(sc in IMPLEMENTED_SELF_CONNECTION_TYPE for sc in x)
        )
    ),
    KEY.INTERACTION_TYPE: lambda x: x in IMPLEMENTED_INTERACTION_TYPE,
    KEY._NORMALIZE_SPH: bool,
    KEY.USE_FLASH_TP: bool,
    KEY.CUEQUIVARIANCE_CONFIG: dict,
    KEY.USE_OEQ: bool,
}


def model_defaults(config):
    defaults = DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG

    if KEY.READOUT_AS_FCN not in config:
        config[KEY.READOUT_AS_FCN] = defaults[KEY.READOUT_AS_FCN]
    if config[KEY.READOUT_AS_FCN] is False:
        defaults.pop(KEY.READOUT_FCN_ACTIVATION, None)
        defaults.pop(KEY.READOUT_FCN_HIDDEN_NEURONS, None)

    return defaults


DEFAULT_DATA_CONFIG = {
    KEY.DTYPE: 'single',
    KEY.DATA_FORMAT: 'ase',
    KEY.DATA_FORMAT_ARGS: {},
    KEY.SAVE_DATASET: False,
    KEY.SAVE_BY_LABEL: False,
    KEY.SAVE_BY_TRAIN_VALID: False,
    KEY.RATIO: 0.0,
    KEY.BATCH_SIZE: 6,
    KEY.PREPROCESS_NUM_CORES: 1,
    KEY.COMPUTE_STATISTICS: True,
    KEY.DATASET_TYPE: 'graph',
    # KEY.USE_SPECIES_WISE_SHIFT_SCALE: False,
    KEY.USE_MODAL_WISE_SHIFT: False,
    KEY.USE_MODAL_WISE_SCALE: False,
    KEY.SHIFT: 'per_atom_energy_mean',
    KEY.SCALE: 'force_rms',
    # KEY.DATA_SHUFFLE: True,
    # KEY.DATA_WEIGHT: False,
    # KEY.DATA_MODALITY: False,
}

DATA_CONFIG_CONDITION = {
    KEY.DTYPE: str,
    KEY.DATA_FORMAT: str,
    KEY.DATA_FORMAT_ARGS: dict,
    KEY.SAVE_DATASET: str,
    KEY.SAVE_BY_LABEL: bool,
    KEY.SAVE_BY_TRAIN_VALID: bool,
    KEY.RATIO: float,
    KEY.BATCH_SIZE: int,
    KEY.PREPROCESS_NUM_CORES: int,
    KEY.DATASET_TYPE: lambda x: x in ['graph', 'atoms'],
    # KEY.USE_SPECIES_WISE_SHIFT_SCALE: bool,
    KEY.SHIFT: lambda x: type(x) in [float, list] or x in IMPLEMENTED_SHIFT,
    KEY.SCALE: lambda x: type(x) in [float, list] or x in IMPLEMENTED_SCALE,
    KEY.USE_MODAL_WISE_SHIFT: bool,
    KEY.USE_MODAL_WISE_SCALE: bool,
    # KEY.DATA_SHUFFLE: bool,
    KEY.COMPUTE_STATISTICS: bool,
    # KEY.DATA_WEIGHT: bool,
    # KEY.DATA_MODALITY: bool,
}


def data_defaults(config):
    defaults = DEFAULT_DATA_CONFIG
    if KEY.LOAD_VALIDSET in config:
        defaults.pop(KEY.RATIO, None)
    return defaults


DEFAULT_TRAINING_CONFIG = {
    KEY.RANDOM_SEED: 1,
    KEY.EPOCH: 300,
    KEY.LOSS: 'mse',
    KEY.LOSS_PARAM: {},
    KEY.OPTIMIZER: 'adam',
    KEY.OPTIM_PARAM: {},
    KEY.SCHEDULER: 'exponentiallr',
    KEY.SCHEDULER_PARAM: {},
    KEY.FORCE_WEIGHT: 0.1,
    KEY.STRESS_WEIGHT: 1e-6,  # SIMPLE-NN default
    KEY.PER_EPOCH: 5,
    # KEY.USE_TESTSET: False,
    KEY.CONTINUE: {
        KEY.CHECKPOINT: False,
        KEY.RESET_OPTIMIZER: False,
        KEY.RESET_SCHEDULER: False,
        KEY.RESET_EPOCH: False,
        KEY.USE_STATISTIC_VALUES_OF_CHECKPOINT: True,
        KEY.USE_STATISTIC_VALUES_FOR_CP_MODAL_ONLY: True,
    },
    # KEY.DEFAULT_MODAL: 'common',
    KEY.CSV_LOG: 'log.csv',
    KEY.NUM_WORKERS: 0,
    KEY.IS_TRAIN_STRESS: True,
    KEY.TRAIN_SHUFFLE: True,
    KEY.ERROR_RECORD: [
        ['Energy', 'RMSE'],
        ['Force', 'RMSE'],
        ['Stress', 'RMSE'],
        ['TotalLoss', 'None'],
    ],
    KEY.BEST_METRIC: 'TotalLoss',
    KEY.USE_WEIGHT: False,
    KEY.USE_MODALITY: False,
}


TRAINING_CONFIG_CONDITION = {
    KEY.RANDOM_SEED: int,
    KEY.EPOCH: int,
    KEY.FORCE_WEIGHT: float,
    KEY.STRESS_WEIGHT: float,
    KEY.USE_TESTSET: None,  # Not used
    KEY.NUM_WORKERS: int,
    KEY.PER_EPOCH: int,
    KEY.CONTINUE: {
        KEY.CHECKPOINT: str,
        KEY.RESET_OPTIMIZER: bool,
        KEY.RESET_SCHEDULER: bool,
        KEY.RESET_EPOCH: bool,
        KEY.USE_STATISTIC_VALUES_OF_CHECKPOINT: bool,
        KEY.USE_STATISTIC_VALUES_FOR_CP_MODAL_ONLY: bool,
    },
    KEY.DEFAULT_MODAL: str,
    KEY.IS_TRAIN_STRESS: bool,
    KEY.TRAIN_SHUFFLE: bool,
    KEY.ERROR_RECORD: error_record_condition,
    KEY.BEST_METRIC: str,
    KEY.CSV_LOG: str,
    KEY.USE_MODALITY: bool,
    KEY.USE_WEIGHT: bool,
}


def train_defaults(config):
    defaults = DEFAULT_TRAINING_CONFIG
    if KEY.IS_TRAIN_STRESS not in config:
        config[KEY.IS_TRAIN_STRESS] = defaults[KEY.IS_TRAIN_STRESS]
    if not config[KEY.IS_TRAIN_STRESS]:
        defaults.pop(KEY.STRESS_WEIGHT, None)
    return defaults
