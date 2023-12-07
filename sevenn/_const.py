from typing import Dict, Any
import os.path
from enum import Enum

import torch

from sevenn.nn.activation import ShiftedSoftPlus
import sevenn._keys as KEY

SEVENN_VERSION = "0.9.0"
IMPLEMENTED_RADIAL_BASIS = ['bessel']
IMPLEMENTED_CUTOFF_FUNCTION = ['poly_cut', 'XPLOR']
SUPPORTING_METRICS = ['RMSE', 'ComponentRMSE', 'MAE', 'Loss']
SUPPORTING_ERROR_TYPES = ['TotalEnergy', 'Energy', 'Force', 'Stress', 'Stress_GPa', 'TotalLoss']


IMPLEMENTED_MODEL = ['E3_equivariant_model']

# string input to real torch function
ACTIVATION = {"relu": torch.nn.functional.relu, "silu": torch.nn.functional.silu,
              "tanh": torch.tanh, "abs": torch.abs, "ssp": ShiftedSoftPlus,
              "sigmoid": torch.sigmoid, "elu": torch.nn.functional.elu,}
ACTIVATION_FOR_EVEN = {"ssp": ShiftedSoftPlus, "silu": torch.nn.functional.silu}
ACTIVATION_FOR_ODD = {"tanh": torch.tanh, "abs": torch.abs}
ACTIVATION_DICT = {"e": ACTIVATION_FOR_EVEN, "o": ACTIVATION_FOR_ODD}


# to avoid torch script to compile torch_geometry.data
AtomGraphDataType = Dict[str, torch.Tensor]


class LossType(Enum):
    ENERGY = 'energy'  # eV or eV/atom
    FORCE = 'force'    # eV/A
    STRESS = 'stress'  # kB

# deprecated
class DataSetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


def is_dir_avail(x):
    return type(x) is str and os.path.isdir(os.path.dirname(x))


def is_file(x):
    return type(x) == str and os.path.isfile(x)


def is_positive(x):
    return x > 0


def is_list_of_file_or_file(x):
    if type(x) is str:
        x = [x]
    return all([os.path.isfile(v) for v in x])

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
            print('w')
            return False
    return True

DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG = {
    KEY.IRREPS_MANUAL: False,
    KEY.NODE_FEATURE_MULTIPLICITY: 32,
    KEY.LMAX: 1,
    KEY.LMAX_EDGE: -1,  # -1 means lmax_edge = lmax
    KEY.LMAX_NODE: -1,  # -1 means lmax_node = lmax
    KEY.IS_PARITY: True,
    KEY.RADIAL_BASIS: {
        KEY.RADIAL_BASIS_NAME: 'bessel',
        KEY.BESSEL_BASIS_NUM: 8,
    },
    KEY.CUTOFF_FUNCTION: {
        KEY.CUTOFF_FUNCTION_NAME: 'poly_cut',
        KEY.POLY_CUT_P: 6,
    },

    KEY.CUTOFF: 4.5,
    KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS: [64, 64],
    KEY.NUM_CONVOLUTION: 3,
    KEY.ACTIVATION_SCARLAR: {"e": "silu", "o": "tanh"},
    KEY.ACTIVATION_GATE: {"e": "silu", "o": "tanh"},
    KEY.AVG_NUM_NEIGHBOR: True,
    KEY.TRAIN_AVG_NUM_NEIGH: False,
    KEY.TRAIN_SHIFT_SCALE: False,
    KEY.OPTIMIZE_BY_REDUCE: False,
    KEY.USE_BIAS_IN_LINEAR: False,
    KEY.READOUT_AS_FCN: False,
    # Applied af readout as fcn is True
    KEY.READOUT_FCN_HIDDEN_NEURONS: [30, 30],
    KEY.READOUT_FCN_ACTIVATION: "relu",
}


DEFAULT_DATA_CONFIG = {
    KEY.DTYPE: "single",
    KEY.DATA_FORMAT: "structure_list",
    KEY.DATA_FORMAT_ARGS: {},
    KEY.FORMAT_OUTPUTS: 'vasp-out',
    #KEY.STRUCTURE_LIST: False,  # deprecated
    KEY.SAVE_DATASET: False,
    KEY.SAVE_BY_LABEL: False,
    KEY.SAVE_BY_TRAIN_VALID: False,
    KEY.RATIO: 0.1,
    KEY.BATCH_SIZE: 6,
    KEY.PREPROCESS_NUM_CORES: 1,
    KEY.USE_SPECIES_WISE_SHIFT_SCALE: False,
    KEY.SHIFT: False,
    KEY.SCALE: False,
    KEY.DATA_SHUFFLE: True,
}


DEFAULT_TRAINING_CONFIG = {
    KEY.RANDOM_SEED: 1,
    KEY.EPOCH: 300,
    KEY.LOSS: 'mse',
    KEY.OPTIMIZER: 'adam',
    KEY.SCHEDULER: 'exponentiallr',
    KEY.FORCE_WEIGHT: 0.1,
    KEY.STRESS_WEIGHT: 1e-6,  # SIMPLE-NN default
    KEY.PER_EPOCH: 5,
    KEY.USE_TESTSET: False,
    KEY.CONTINUE: {
        # KEY.CHECKPOINT: False, > handled manually in parse_input
        KEY.RESET_OPTIMIZER: False,
        KEY.RESET_SCHEDULER: False,
        KEY.RESET_EPOCH: False,
    },
    KEY.CSV_LOG: "log.csv",
    KEY.NUM_WORKERS: 0,
    KEY.IS_TRACE_STRESS: False,
    KEY.IS_TRAIN_STRESS: True,
    KEY.TRAIN_SHUFFLE: True,
    KEY.ERROR_RECORD: [
        ["Energy", "RMSE"],
        ["Force", "RMSE"],
        ["Stress", "RMSE"],
        ["TotalLoss", "None"],
    ],
    KEY.BEST_METRIC: "TotalLoss",
}


# TODO: implement 'if provided, use condition' case for inputs
# condition for each inputs, key omitted here should be initialized by hand
MODEL_CONFIG_CONDITION = {
    KEY.NODE_FEATURE_MULTIPLICITY: is_positive,
    KEY.LMAX: lambda x: x >= 0,
    KEY.LMAX_EDGE: lambda x: x >= -1,
    KEY.LMAX_NODE: lambda x: x >= -1,
    KEY.IS_PARITY: None,
    # TODO: change these param inputs same as optimizer, scheduler, and loss
    KEY.RADIAL_BASIS: {
        KEY.RADIAL_BASIS_NAME: lambda x: x in IMPLEMENTED_RADIAL_BASIS,
    },
    KEY.CUTOFF_FUNCTION: {
        KEY.CUTOFF_FUNCTION_NAME: lambda x: x in IMPLEMENTED_CUTOFF_FUNCTION,
    },
    KEY.CUTOFF: is_positive,
    KEY.NUM_CONVOLUTION: is_positive,
    KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS:
        lambda x: all(val > 0 and isinstance(val, int) for val in x),
    KEY.TRAIN_SHIFT_SCALE: None,
    KEY.TRAIN_AVG_NUM_NEIGH: None,
    KEY.OPTIMIZE_BY_REDUCE: None,
    KEY.USE_BIAS_IN_LINEAR: None,
    KEY.READOUT_AS_FCN: None,
    KEY.READOUT_FCN_HIDDEN_NEURONS:
        lambda x: all(val > 0 and isinstance(val, int) for val in x),
    KEY.READOUT_FCN_ACTIVATION: lambda x: x in ACTIVATION.keys(),
}


DATA_CONFIG_CONDITION = {
    KEY.DTYPE: lambda x: x.lower() in ["single", "double"],
    KEY.DATA_FORMAT: lambda x: x in ["structure_list", "ase", "pkl", "pickle"],
    KEY.DATA_FORMAT_ARGS: lambda x: type(x) is dict,
    KEY.FORMAT_OUTPUTS: lambda x: x in ["vasp-out", "vasp", "vasp-xdatcar"],
    KEY.SAVE_DATASET: None,
    KEY.SAVE_BY_LABEL: None,
    KEY.SAVE_BY_TRAIN_VALID: None,
    KEY.RATIO: lambda x: type(x) is float and x > 0.0 and x < 0.5,
    KEY.BATCH_SIZE: is_positive,
    KEY.PREPROCESS_NUM_CORES: is_positive,
    KEY.USE_SPECIES_WISE_SHIFT_SCALE: None,
    KEY.DATA_SHUFFLE: None,
}

TRAINING_CONFIG_CONDITION = {
    KEY.RANDOM_SEED: is_positive,
    KEY.EPOCH: is_positive,
    KEY.FORCE_WEIGHT: is_positive,
    KEY.STRESS_WEIGHT: lambda x: x is None or is_positive(x),
    KEY.USE_TESTSET: None,
    KEY.NUM_WORKERS: is_positive,
    KEY.PER_EPOCH: lambda x: is_positive(x) or x is False,
    KEY.CONTINUE: {
        KEY.CHECKPOINT: None,
        KEY.RESET_OPTIMIZER: None,
        KEY.RESET_SCHEDULER: None,
        KEY.RESET_EPOCH: None,
    },
    KEY.IS_TRACE_STRESS: None,
    KEY.IS_TRAIN_STRESS: None,
    KEY.TRAIN_SHUFFLE: None,
    KEY.ERROR_RECORD: error_record_condition,
    KEY.BEST_METRIC: lambda x: type(x) is str and x in ["Energy", "Force", "Stress", "TotalLoss"],
    KEY.CSV_LOG: lambda x: type(x) is str,
}
