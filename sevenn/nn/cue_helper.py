import itertools
import warnings
from typing import Iterator, Literal, Union

import e3nn.o3 as o3
import numpy as np

from .convolution import IrrepsConvolution
from .linear import IrrepsLinear
from .self_connection import SelfConnectionIntro, SelfConnectionLinearIntro

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    _CUE_AVAILABLE = True

    # Obatained from MACE
    class O3_e3nn(cue.O3):
        def __mul__(  # type: ignore
            rep1: 'O3_e3nn', rep2: 'O3_e3nn'
        ) -> Iterator['O3_e3nn']:
            return [  # type: ignore
                O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)
            ]

        @classmethod
        def clebsch_gordan(  # type: ignore
            cls, rep1: 'O3_e3nn', rep2: 'O3_e3nn', rep3: 'O3_e3nn'
        ) -> np.ndarray:
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

            if rep1.p * rep2.p == rep3.p:
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                    rep3.dim
                )
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        def __lt__(  # type: ignore
            rep1: 'O3_e3nn', rep2: 'O3_e3nn'
        ) -> bool:
            rep2 = rep1._from(rep2)  # type: ignore
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls) -> Iterator['O3_e3nn']:
            for l in itertools.count(0):
                yield O3_e3nn(l=l, p=1 * (-1) ** l)
                yield O3_e3nn(l=l, p=-1 * (-1) ** l)

except ImportError:
    _CUE_AVAILABLE = False


def is_cue_available():
    return _CUE_AVAILABLE


def cue_needed(func):
    def wrapper(*args, **kwargs):
        if is_cue_available():
            return func(*args, **kwargs)
        else:
            raise ImportError('cue is not available')

    return wrapper


def _check_may_not_compatible(orig_kwargs, defaults):
    for k, v in defaults.items():
        v_given = orig_kwargs.pop(k, v)
        if v_given != v:
            warnings.warn(f'{k}: {v} is ignored to use cuEquivariance')


def is_cue_cuda_available_model(config):
    if config.get('use_bias_in_linear', False):
        warnings.warn('Bias in linear can not be used with cueq, fallback to e3nn')
        return False
    else:
        return True


@cue_needed
def as_cue_irreps(irreps: o3.Irreps, group: Literal['SO3', 'O3']):
    """Convert e3nn irreps to given group's cue irreps"""
    if group == 'SO3':
        assert all(irrep.ir.p == 1 for irrep in irreps)
        return cue.Irreps('SO3', str(irreps).replace('e', ''))  # type: ignore
    elif group == 'O3':
        return cue.Irreps(O3_e3nn, str(irreps))  # type: ignore
    else:
        raise ValueError(f'Unknown group: {group}')


@cue_needed
def patch_linear(
    module: Union[IrrepsLinear, SelfConnectionLinearIntro],
    group: Literal['SO3', 'O3'],
    **cue_kwargs,
):
    assert not module.layer_instantiated

    module.irreps_in = as_cue_irreps(module.irreps_in, group)  # type: ignore
    module.irreps_out = as_cue_irreps(module.irreps_out, group)  # type: ignore

    orig_kwargs = module.linear_kwargs

    may_not_compatible_default = dict(
        f_in=None,
        f_out=None,
        instructions=None,
        biases=False,
        path_normalization='element',
        _optimize_einsums=None,
    )
    # pop may_not_compatible_defaults
    _check_may_not_compatible(orig_kwargs, may_not_compatible_default)

    module.linear_cls = cuet.Linear  # type: ignore
    orig_kwargs.update(**cue_kwargs)
    return module


@cue_needed
def patch_convolution(
    module: IrrepsConvolution,
    group: Literal['SO3', 'O3'],
    **cue_kwargs,
):
    assert not module.layer_instantiated

    # conv_kwargs will be patched in place
    conv_kwargs = module.convolution_kwargs
    conv_kwargs.update(
        dict(
            irreps_in1=as_cue_irreps(conv_kwargs.get('irreps_in1'), group),
            irreps_in2=as_cue_irreps(conv_kwargs.get('irreps_in2'), group),
            filter_irreps_out=as_cue_irreps(conv_kwargs.pop('irreps_out'), group),
        )
    )

    inst_orig = conv_kwargs.pop('instructions')
    inst_sorted = sorted(inst_orig, key=lambda x: x[2])
    assert all([a == b for a, b in zip(inst_orig, inst_sorted)])

    may_not_compatible_default = dict(
        in1_var=None,
        in2_var=None,
        out_var=None,
        irrep_normalization=False,
        path_normalization='element',
        compile_left_right=True,
        compile_right=False,
        _specialized_code=None,
        _optimize_einsums=None,
    )
    # pop may_not_compatible_defaults
    _check_may_not_compatible(conv_kwargs, may_not_compatible_default)

    module.convolution_cls = cuet.ChannelWiseTensorProduct  # type: ignore
    conv_kwargs.update(**cue_kwargs)
    return module


@cue_needed
def patch_fully_connected(
    module: SelfConnectionIntro,
    group: Literal['SO3', 'O3'],
    **cue_kwargs,
):
    assert not module.layer_instantiated

    module.irreps_in1 = as_cue_irreps(module.irreps_in1, group)  # type: ignore
    module.irreps_in2 = as_cue_irreps(module.irreps_in2, group)  # type: ignore
    module.irreps_out = as_cue_irreps(module.irreps_out, group)  # type: ignore

    may_not_compatible_default = dict(
        irrep_normalization=None,
        path_normalization=None,
    )
    # pop may_not_compatible_defaults
    _check_may_not_compatible(
        module.fc_tensor_product_kwargs, may_not_compatible_default
    )

    module.fc_tensor_product_cls = cuet.FullyConnectedTensorProduct  # type: ignore
    module.fc_tensor_product_kwargs.update(**cue_kwargs)
    return module
