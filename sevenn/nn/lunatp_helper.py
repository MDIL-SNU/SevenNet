import torch

from .convolution import IrrepsConvolution, IrrepsScatterGatterFusedConvolution

try:
    from LunaEquivariance import TensorProductConv, make_graph

    _LUNATP_AVAILABLE = True
except ImportError:
    _LUNATP_AVAILABLE = False


def is_lunatp_available() -> bool:
    return _LUNATP_AVAILABLE and torch.cuda.is_available()


def lunatp_needed(func):
    def wrapper(*args, **kwargs):
        if is_lunatp_available():
            return func(*args, **kwargs)
        raise ImportError('LunaTP is not available (no GPU found or import fail)')

    return wrapper


class LunaTPConvolution(torch.nn.Module):
    """
    Wrapper around LunaEquivariance.TensorProductConv to match
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
        if not is_lunatp_available():
            raise ImportError('LunaTP is not available')
        self.tp_conv = TensorProductConv(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )

    def forward(self, x, edge_filter, weight, edge_src, edge_dst):
        # LunaTP requires edges sorted by destination
        if not bool(torch.all(edge_dst[1:] >= edge_dst[:-1])):
            order = torch.argsort(edge_dst, stable=True)
            edge_src, edge_dst = edge_src[order], edge_dst[order]
            edge_filter, weight = edge_filter[order], weight[order]
        # TODO: build the graph once per forward, not once per layer
        graph = make_graph(edge_src, edge_dst, x.shape[0])
        return self.tp_conv(x, edge_filter, weight, graph=graph)


@lunatp_needed
def patch_convolution(irreps_convolution: IrrepsConvolution):
    assert not irreps_convolution.layer_instantiated, (
        'Convolution layer already instantiated; cannot patch'
    )
    ret = IrrepsScatterGatterFusedConvolution.from_irreps_convolution(
        irreps_convolution
    )
    ret.convolution_cls = LunaTPConvolution
    return ret
