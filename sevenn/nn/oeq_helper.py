import torch

from .convolution import IrrepsConvolution, IrrepsScatterGatterFusedConvolution

try:
    from openequivariance import (
        TensorProductConv,
        TPProblem,
        torch_to_oeq_dtype,
    )

    _OEQ_AVAILABLE = True
except ImportError:
    _OEQ_AVAILABLE = False


def is_oeq_available() -> bool:
    return _OEQ_AVAILABLE and torch.cuda.is_available()


def oeq_needed(func):
    def wrapper(*args, **kwargs):
        if is_oeq_available():
            return func(*args, **kwargs)
        raise ImportError('OpenEquivariance (oeq) is not available')

    return wrapper


class OEQConvolution(torch.nn.Module):
    """
    Wrapper around openequivariance.TensorProductConv to match
    IrrepsScatterGatterFusedConvolution.convolution_cls interface.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights: bool = False,
        internal_weights: bool = False,
    ):
        super().__init__()
        if not is_oeq_available():
            raise ImportError('OpenEquivariance is not available')
        self.dtype = torch.get_default_dtype()
        tpp = TPProblem(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            irrep_dtype=torch_to_oeq_dtype(self.dtype),
            weight_dtype=torch_to_oeq_dtype(self.dtype),
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        self.tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def forward(self, x, edge_filter, weight, edge_src, edge_dst):
        # OEQ rows=dst, cols=src (swapped vs sevennet's arg order)
        # OEQ needs int64; sevennet casts to int32 before convolution
        return self.tp_conv(
            x.to(self.dtype),
            edge_filter.to(self.dtype),
            weight.to(self.dtype),
            edge_dst.to(torch.int64),  # rows = dst
            edge_src.to(torch.int64),  # cols = src
        )


@oeq_needed
def patch_convolution(irreps_convolution: IrrepsConvolution):

    assert not irreps_convolution.layer_instantiated, (
        'Convolution layer already instantiated; cannot patch'
    )
    ret = IrrepsScatterGatterFusedConvolution.from_irreps_convolution(
        irreps_convolution
    )
    ret.convolution_cls = OEQConvolution
    return ret
