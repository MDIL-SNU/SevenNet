import warnings
from collections import OrderedDict
from typing import Union

from e3nn.o3 import FullTensorProduct, Irreps

import sevenn._const as _const
import sevenn._keys as KEY
from .nn.edge_embedding import (
    BesselBasis,
    EdgeEmbedding,
    EdgePreprocess,
    PolynomialCutoff,
    SphericalEncoding,
    XPLORCutoff,
)
from .nn.force_output import ForceStressOutput
from .nn.linear import AtomReduce, FCN_e3nn, IrrepsLinear
from .nn.node_embedding import OnehotEmbedding
from .nn.scale import Rescale, SpeciesWiseRescale
from .nn.interaction_blocks import _NequIP_interaction_block
from .nn.sequential import AtomGraphSequential

# warning from PyTorch, about e3nn type annotations
warnings.filterwarnings(
    'ignore',
    message=(
        "The TorchScript type system doesn't "
        "support instance-level annotations"
    ),
)


def infer_irreps_out(
    irreps_x: Irreps,
    irreps_operand: Irreps,
    drop_l: Union[bool, int] = False,
    fix_multiplicity: Union[bool, int] = False,
    only_even_p: bool = False,
):
    # (mul, (ir, p))
    irreps_out = FullTensorProduct(
        irreps_x, irreps_operand
    ).irreps_out.simplify()
    new_irreps_elem = []
    for mul, (l, p) in irreps_out:
        elem = (mul, (l, p))
        if drop_l is not False and l > drop_l:
            continue
        if only_even_p and p == -1:
            continue
        if fix_multiplicity:
            elem = (fix_multiplicity, (l, p))
        new_irreps_elem.append(elem)
    return Irreps(new_irreps_elem)


def init_radial_basis(config):
    radial_basis_dct = config[KEY.RADIAL_BASIS]
    cutoff = config[KEY.CUTOFF]

    if radial_basis_dct[KEY.RADIAL_BASIS_NAME] == 'bessel':
        basis_num = radial_basis_dct[KEY.BESSEL_BASIS_NUM]
        return BesselBasis(basis_num, cutoff), basis_num

    raise RuntimeError('something went very wrong...')


# TODO: totally messed up :(
def init_cutoff_function(config):
    cutoff_function_dct = config[KEY.CUTOFF_FUNCTION]
    cutoff = config[KEY.CUTOFF]
    if cutoff_function_dct[KEY.CUTOFF_FUNCTION_NAME] == 'poly_cut':
        p = cutoff_function_dct[KEY.POLY_CUT_P]
        return PolynomialCutoff(p, cutoff)
    elif cutoff_function_dct[KEY.CUTOFF_FUNCTION_NAME] == 'XPLOR':
        return XPLORCutoff(cutoff_function_dct['cutoff_on'], cutoff)

    raise RuntimeError('something went very wrong...')


def _to_parallel_model(layers: OrderedDict, config):
    layers = list(layers.items())
    layers_list = []

    num_species = config[KEY.NUM_SPECIES]
    one_hot_irreps = Irreps(f'{num_species}x0e')
    feature_multiplicity = config[KEY.NODE_FEATURE_MULTIPLICITY]
    irreps_manual = None
    is_parity = config[KEY.IS_PARITY]  # boolean
    lmax = config[KEY.LMAX]
    lmax_edge = config[KEY.LMAX_EDGE] if config[KEY.LMAX_EDGE] >= 0 else lmax
    num_convolution_layer = config[KEY.NUM_CONVOLUTION]

    if config[KEY.IRREPS_MANUAL] is not False:
        irreps_manual = config[KEY.IRREPS_MANUAL]
        try:
            irreps_manual = [Irreps(irr) for irr in irreps_manual]
            assert len(irreps_manual) == num_convolution_layer + 1
        except Exception:
            raise RuntimeError('invalid irreps_manual input given')

    irreps_node_zero = (
        Irreps(f'{feature_multiplicity}x0e')
        if irreps_manual is None
        else irreps_manual[0]
    )

    def insert_after(module_name_after, key_module_pair, layers):
        idx = -1
        for i, (key, _) in enumerate(layers):
            if key == module_name_after:
                idx = i
                break
        if idx == -1:
            assert False
        layers.insert(idx + 1, key_module_pair)
        return layers

    def slice_until_this(module_name, layers):
        idx = -1
        for i, (key, _) in enumerate(layers):
            if key == module_name:
                idx = i
                break
        first_to = layers[:idx+1]
        remain = layers[idx+1:]
        return first_to, remain


    layers = insert_after('onehot_to_feature_x', (
        'one_hot_ghost', OnehotEmbedding(
             data_key_x=KEY.NODE_FEATURE_GHOST,
             num_classes=num_species,
             data_key_save=None,
             data_key_additional=None,
         )), layers
    )
    layers = insert_after('one_hot_ghost', (
        'ghost_onehot_to_feature_x', IrrepsLinear(
            irreps_in=one_hot_irreps,
            irreps_out=irreps_node_zero,
            data_key_in=KEY.NODE_FEATURE_GHOST,
            biases=config[KEY.USE_BIAS_IN_LINEAR],
         )), layers
    )
    layers = insert_after('0_self_interaction_1', (
        'ghost_0_self_interaction_1', IrrepsLinear(
            irreps_node_zero, irreps_node_zero,
            data_key_in=KEY.NODE_FEATURE_GHOST,
            biases=config[KEY.USE_BIAS_IN_LINEAR],
        )), layers
    )
    # assign modules (before first communications)
    # initialize edge related to retain position gradients
    for i in range(1, num_convolution_layer):
        sliced, layers =\
            slice_until_this(f'{i}_self_interaction_1', layers)
        layers_list.append(OrderedDict(sliced))
        radial_basis, _ = init_radial_basis(config)
        cutoff_function = init_cutoff_function(config)
        sph_encode = SphericalEncoding(lmax_edge, -1 if is_parity else 1)
        layers.insert(0, (
            'edge_embedding', EdgeEmbedding(
                basis_module=radial_basis,
                cutoff_module=cutoff_function,
                spherical_module=sph_encode,
            )
        ))

    layers_list.append(OrderedDict(layers))
    del layers_list[0]['edge_preprocess'] # done in LAMMPS
    del layers_list[-1]["force_output"] # done in LAMMPS
    return layers_list


# TODO: it gets bigger and bigger. refactor it
def build_E3_equivariant_model(model_config: dict, parallel=False):
    layers = OrderedDict()

    ################## initialize ####################
    cutoff = model_config[KEY.CUTOFF]
    num_species = model_config[KEY.NUM_SPECIES]
    feature_multiplicity = model_config[KEY.NODE_FEATURE_MULTIPLICITY]
    num_convolution_layer = model_config[KEY.NUM_CONVOLUTION]
    is_parity = model_config[KEY.IS_PARITY]  # boolean

    lmax_node = lmax_edge = model_config[KEY.LMAX]
    if model_config[KEY.LMAX_EDGE] > 0:
        lmax_edge = model_config[KEY.LMAX_EDGE]
    if model_config[KEY.LMAX_EDGE] > 0:
        lmax_node = model_config[KEY.LMAX_EDGE]

    irreps_manual = None
    if model_config[KEY.IRREPS_MANUAL] is not False:
        irreps_manual = model_config[KEY.IRREPS_MANUAL]
        try:
            irreps_manual = [Irreps(irr) for irr in irreps_manual]
            assert len(irreps_manual) == num_convolution_layer + 1
        except Exception:
            raise RuntimeError('invalid irreps_manual input')

    radial_basis_module, radial_basis_num = init_radial_basis(model_config)
    cutoff_function_module = init_cutoff_function(model_config)

    avg_num_neigh = model_config[KEY.AVG_NUM_NEIGH]
    if type(avg_num_neigh) is not list:
        avg_num_neigh = [avg_num_neigh] * num_convolution_layer

    use_bias_in_linear = model_config[KEY.USE_BIAS_IN_LINEAR]

    edge_embedding = EdgeEmbedding(
        basis_module=radial_basis_module,
        cutoff_module=cutoff_function_module,
        spherical_module=SphericalEncoding(lmax_edge, -1 if is_parity else 1),
    )
    irreps_filter = edge_embedding.spherical.irreps_out
    layers.update({'edge_preprocess': EdgePreprocess(is_stress=True)})
    layers.update({'edge_embedding': edge_embedding})

    one_hot_irreps = Irreps(f'{num_species}x0e')
    irreps_x = (
        Irreps(f'{feature_multiplicity}x0e')
        if irreps_manual is None
        else irreps_manual[0]
    )
    layers.update({
        'onehot_idx_to_onehot': OnehotEmbedding(num_classes=num_species),
        'onehot_to_feature_x': IrrepsLinear(
            irreps_in=one_hot_irreps,
            irreps_out=irreps_x,
            data_key_in=KEY.NODE_FEATURE,
            biases=use_bias_in_linear,
        ),
    })

    weight_nn_hidden = model_config[KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS]
    weight_nn_layers = [radial_basis_num] + weight_nn_hidden

    for t in range(num_convolution_layer):
        only_even_p = False
        if t == num_convolution_layer - 1:
            lmax_node = 0
            only_even_p = True

        # irreps out after tensorproduct
        irreps_out_tp = infer_irreps_out(
            irreps_x, irreps_filter,
            drop_l=lmax_node, only_even_p=only_even_p,
        )

        # node irreps out
        irreps_out =\
            infer_irreps_out(
                irreps_x, irreps_filter,
                drop_l=lmax_node, only_even_p=only_even_p,
                fix_multiplicity=feature_multiplicity,
            ) if irreps_manual is None else irreps_manual[t + 1]

        interaction_block = _NequIP_interaction_block(
            model_config,
            irreps_x,
            irreps_filter,
            irreps_out_tp,
            irreps_out,
            weight_nn_layers,
            avg_num_neigh[t],
            t,
            parallel
        )
        layers.update(interaction_block)
        irreps_x = irreps_out
        # end of interaction block for-loop

    if model_config[KEY.READOUT_AS_FCN] is False:
        mid_dim = (
            feature_multiplicity
            if irreps_manual is None
            else irreps_manual[-1].num_irreps
        )
        hidden_irreps = Irreps([(mid_dim // 2, (0, 1))])
        layers.update({
            'reduce_input_to_hidden': IrrepsLinear(
                irreps_x,
                hidden_irreps,
                data_key_in=KEY.NODE_FEATURE,
                biases=use_bias_in_linear,
            ),
            'reduce_hidden_to_energy': IrrepsLinear(
                hidden_irreps,
                Irreps([(1, (0, 1))]),
                data_key_in=KEY.NODE_FEATURE,
                data_key_out=KEY.SCALED_ATOMIC_ENERGY,
                biases=use_bias_in_linear,
            ),
        })
    else:
        act = _const.ACTIVATION[model_config[KEY.READOUT_FCN_ACTIVATION]]
        hidden_neurons = model_config[KEY.READOUT_FCN_HIDDEN_NEURONS]
        layers.update({
            'readout_FCN': FCN_e3nn(
                dim_out=1,
                hidden_neurons=hidden_neurons,
                activation=act,
                data_key_in=KEY.NODE_FEATURE,
                data_key_out=KEY.SCALED_ATOMIC_ENERGY,
                irreps_in=irreps_x,
            )
        })

    shift = model_config[KEY.SHIFT]
    scale = model_config[KEY.SCALE]
    train_shift_scale = model_config[KEY.TRAIN_SHIFT_SCALE]
    rescale_module = (
        SpeciesWiseRescale
        if model_config[KEY.USE_SPECIES_WISE_SHIFT_SCALE]
        else Rescale
    )
    layers.update({
        'rescale_atomic_energy': rescale_module(
            shift=shift,
            scale=scale,
            data_key_in=KEY.SCALED_ATOMIC_ENERGY,
            data_key_out=KEY.ATOMIC_ENERGY,
            train_shift_scale=train_shift_scale,
        ),
        'reduce_total_enegy': AtomReduce(
            data_key_in=KEY.ATOMIC_ENERGY,
            data_key_out=KEY.PRED_TOTAL_ENERGY,
            constant=1.0,
        ),
    })
    gradient_module = ForceStressOutput(
        data_key_energy=KEY.PRED_TOTAL_ENERGY,
        data_key_force=KEY.PRED_FORCE,
        data_key_stress=KEY.PRED_STRESS,
    )
    layers.update({'force_output': gradient_module})

    # output extraction part
    type_map = model_config[KEY.TYPE_MAP]
    if parallel:
        layers_list = _to_parallel_model(layers, model_config)
        return [AtomGraphSequential(v, cutoff, type_map) for v in layers_list]
    else:
        return AtomGraphSequential(layers, cutoff, type_map)

