import warnings
from collections import OrderedDict

from e3nn.o3 import Irreps

import sevenn._const as _const
import sevenn._keys as KEY
import sevenn.util as util
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
from .nn.self_connection import (
    SelfConnectionIntro,
    SelfConnectionLinearIntro,
    SelfConnectionOutro,
)
from .nn.interaction_blocks import (
    NequIP_interaction_block,
    MACE_interaction_block,
)
from .nn.sequential import AtomGraphSequential

# warning from PyTorch, about e3nn type annotations
warnings.filterwarnings(
    'ignore',
    message=(
        "The TorchScript type system doesn't "
        "support instance-level annotations"
    ),
)


def init_self_connection(config):
    self_connection_type = config[KEY.SELF_CONNECTION_TYPE]
    intro, outro = None, None
    if self_connection_type == 'none':
        pass
    elif self_connection_type == 'nequip':
        intro, outro = SelfConnectionIntro, SelfConnectionOutro
        return SelfConnectionIntro, SelfConnectionOutro
    elif self_connection_type == 'linear':
        intro, outro = SelfConnectionLinearIntro, SelfConnectionOutro
    else:
        raise ValueError('something went wrong...')
    return intro, outro


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
    num_classes = layers['onehot_idx_to_onehot'].num_classes
    one_hot_irreps = Irreps(f'{num_classes}x0e')
    irreps_node_zero = layers['onehot_to_feature_x'].linear.irreps_out

    layers = list(layers.items())
    layers_list = []

    is_parity = config[KEY.IS_PARITY]  # boolean
    lmax = config[KEY.LMAX]
    lmax_edge = config[KEY.LMAX_EDGE] if config[KEY.LMAX_EDGE] > 0 else lmax
    num_convolution_layer = config[KEY.NUM_CONVOLUTION]

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
             num_classes=num_classes,
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
def build_E3_equivariant_model(config: dict, parallel=False):
    layers = OrderedDict()

    ################## initialize ####################
    cutoff = config[KEY.CUTOFF]
    num_species = config[KEY.NUM_SPECIES]
    feature_multiplicity = config[KEY.NODE_FEATURE_MULTIPLICITY]
    num_convolution_layer = config[KEY.NUM_CONVOLUTION]
    is_parity = config[KEY.IS_PARITY]  # boolean
    interaction_type = config[KEY.INTERACTION_TYPE]
    use_bias_in_linear = config[KEY.USE_BIAS_IN_LINEAR]

    lmax_node = lmax_edge = config[KEY.LMAX]
    if config[KEY.LMAX_EDGE] > 0:
        lmax_edge = config[KEY.LMAX_EDGE]
    if config[KEY.LMAX_NODE] > 0:
        lmax_node = config[KEY.LMAX_NODE]

    act_radial = _const.ACTIVATION[config[KEY.ACTIVATION_RADIAL]]
    self_connection_pair = init_self_connection(config)

    irreps_manual = None
    if config[KEY.IRREPS_MANUAL] is not False:
        irreps_manual = config[KEY.IRREPS_MANUAL]
        try:
            irreps_manual = [Irreps(irr) for irr in irreps_manual]
            assert len(irreps_manual) == num_convolution_layer + 1
        except Exception:
            raise RuntimeError('invalid irreps_manual input')

    radial_basis_module, radial_basis_num = init_radial_basis(config)
    cutoff_function_module = init_cutoff_function(config)

    avg_num_neigh = config[KEY.AVG_NUM_NEIGH]
    if type(avg_num_neigh) is not list:
        avg_num_neigh = [avg_num_neigh] * num_convolution_layer
    train_conv_denominator = config[KEY.TRAIN_AVG_NUM_NEIGH]

    #-----------------------model definition-------------------------#
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
        if irreps_manual is None else irreps_manual[0]
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

    weight_nn_hidden = config[KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS]
    weight_nn_layers = [radial_basis_num] + weight_nn_hidden

    param_interaction_block = {
        "irreps_filter": irreps_filter,
        "weight_nn_layers": weight_nn_layers,
        "conv_denominator": avg_num_neigh,
        "train_conv_denominator": train_conv_denominator,
        "self_connection_pair": self_connection_pair,
        "act_radial": act_radial,
        "bias_in_linear": use_bias_in_linear,
        "num_species": num_species,
        "parallel": parallel,
    }
    interaction_builder = None
    if interaction_type == 'nequip':
        act_scalar = {}
        act_gate = {}
        for k, v in config[KEY.ACTIVATION_SCARLAR].items():
            act_scalar[k] = _const.ACTIVATION_DICT[k][v]
        for k, v in config[KEY.ACTIVATION_GATE].items():
            act_gate[k] = _const.ACTIVATION_DICT[k][v]
        param_interaction_block.update({
            "act_scalar": act_scalar,
            "act_gate": act_gate,
        })
        interaction_builder = NequIP_interaction_block
    elif interaction_type == 'mace':
        param_interaction_block.update({
            "correlation": config[KEY.CORRELATION],
        })
        interaction_builder = MACE_interaction_block


    for t in range(num_convolution_layer):
        param_interaction_block.update({
            "irreps_x": irreps_x, "t": t, 
            "conv_denominator": avg_num_neigh[t]
        })

        if interaction_type == 'nequip':
            parity_mode = "full"
            if t == num_convolution_layer - 1:
                lmax_node = 0
                parity_mode = "even"
            irreps_out_tp = util.infer_irreps_out(
                irreps_x, irreps_filter, lmax_node, parity_mode,
            )
        elif interaction_type == 'mace':
            parity_mode = "sph"
            irreps_out_tp = util.infer_irreps_out(
                irreps_x, irreps_filter, lmax_edge, parity_mode,
            )
            if t == num_convolution_layer - 1:  # scalar output
                lmax_node = 0
                parity_mode = "even"

        irreps_out = util.infer_irreps_out(
            irreps_x, irreps_filter, lmax_node, parity_mode,
            fix_multiplicity=feature_multiplicity,
        ) if irreps_manual is None else irreps_manual[t + 1]

        param_interaction_block.update({
            "irreps_out_tp": irreps_out_tp,
            "irreps_out": irreps_out,
        })
        layers.update(interaction_builder(**param_interaction_block))
        irreps_x = irreps_out


    if config[KEY.READOUT_AS_FCN] is False:
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
        act = _const.ACTIVATION[config[KEY.READOUT_FCN_ACTIVATION]]
        hidden_neurons = config[KEY.READOUT_FCN_HIDDEN_NEURONS]
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

    shift = config[KEY.SHIFT]
    scale = config[KEY.SCALE]
    train_shift_scale = config[KEY.TRAIN_SHIFT_SCALE]
    rescale_module = (
        SpeciesWiseRescale
        if config[KEY.USE_SPECIES_WISE_SHIFT_SCALE]
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
    type_map = config[KEY.TYPE_MAP]
    if parallel:
        layers_list = _to_parallel_model(layers, config)
        return [AtomGraphSequential(v, cutoff, type_map) for v in layers_list]
    else:
        return AtomGraphSequential(layers, cutoff, type_map)

