import warnings
from collections import OrderedDict

from e3nn.o3 import Irreps

import sevenn._const as _const
import sevenn._keys as KEY
from sevenn.nn.convolution import IrrepsConvolution
import sevenn.util as util
from sevenn.nn.edge_embedding import (
    BesselBasis,
    EdgeEmbedding,
    EdgePreprocess,
    PolynomialCutoff,
    SphericalEncoding,
    XPLORCutoff,
)
from sevenn.nn.equivariant_gate import EquivariantGate
from sevenn.nn.force_output import ForceOutputFromEdge, ForceStressOutput
from sevenn.nn.linear import AtomReduce, FCN_e3nn, IrrepsLinear
from sevenn.nn.node_embedding import OnehotEmbedding
from sevenn.nn.scale import Rescale, SpeciesWiseRescale
from sevenn.nn.self_connection import (
    SelfConnectionIntro,
    SelfConnectionLinearIntro,
    SelfConnectionOutro,
)
from sevenn.nn.sequential import AtomGraphSequential

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
    param = {"cutoff_length": config[KEY.CUTOFF]}
    param.update(radial_basis_dct)
    del param[KEY.RADIAL_BASIS_NAME]

    if radial_basis_dct[KEY.RADIAL_BASIS_NAME] == 'bessel':
        basis_function =  BesselBasis(**param)
        return basis_function, basis_function.num_basis

    raise RuntimeError('something went very wrong...')


def init_cutoff_function(config):
    cutoff_function_dct = config[KEY.CUTOFF_FUNCTION]
    param = {"cutoff_length": config[KEY.CUTOFF]}
    param.update(cutoff_function_dct)
    del param[KEY.CUTOFF_FUNCTION_NAME]

    if cutoff_function_dct[KEY.CUTOFF_FUNCTION_NAME] == 'poly_cut':
        return PolynomialCutoff(**param)
    elif cutoff_function_dct[KEY.CUTOFF_FUNCTION_NAME] == 'XPLOR':
        return XPLORCutoff(**param)
    raise RuntimeError('something went very wrong...')


# TODO: it gets bigger and bigger. refactor it
def build_E3_equivariant_model(config: dict, parallel=False):
    """
    IDENTICAL to nequip model
    atom embedding is not part of model
    """
    data_key_weight_input = KEY.EDGE_EMBEDDING  # default

    # parameter initialization
    cutoff = config[KEY.CUTOFF]
    num_species = config[KEY.NUM_SPECIES]

    feature_multiplicity = config[KEY.NODE_FEATURE_MULTIPLICITY]

    lmax = config[KEY.LMAX]
    lmax_edge = (
        config[KEY.LMAX_EDGE]
        if config[KEY.LMAX_EDGE] >= 0
        else lmax
    )
    lmax_node = (
        config[KEY.LMAX_NODE]
        if config[KEY.LMAX_NODE] >= 0
        else lmax
    )

    num_convolution_layer = config[KEY.NUM_CONVOLUTION]

    is_parity = config[KEY.IS_PARITY]  # boolean

    irreps_spherical_harm = Irreps.spherical_harmonics(
        lmax_edge, -1 if is_parity else 1
    )
    if parallel:
        layers_list = [OrderedDict() for _ in range(num_convolution_layer)]
        layers_idx = 0
        layers = layers_list[0]
    else:
        layers = OrderedDict()

    irreps_manual = None
    if config[KEY.IRREPS_MANUAL] is not False:
        irreps_manual = config[KEY.IRREPS_MANUAL]
        try:
            irreps_manual = [Irreps(irr) for irr in irreps_manual]
            assert len(irreps_manual) == num_convolution_layer + 1
        except Exception:
            raise RuntimeError('invalid irreps_manual input given')

    sc_intro, sc_outro = init_self_connection(config)

    act_scalar = {}
    act_gate = {}
    for k, v in config[KEY.ACTIVATION_SCARLAR].items():
        act_scalar[k] = _const.ACTIVATION_DICT[k][v]
    for k, v in config[KEY.ACTIVATION_GATE].items():
        act_gate[k] = _const.ACTIVATION_DICT[k][v]
    act_radial = _const.ACTIVATION[config[KEY.ACTIVATION_RADIAL]]

    radial_basis_module, radial_basis_num = init_radial_basis(config)
    cutoff_function_module = init_cutoff_function(config)

    conv_denominator = config[KEY.CONV_DENOMINATOR]
    if not isinstance(conv_denominator, list):
        conv_denominator = [conv_denominator] * num_convolution_layer
    train_conv_denominator = config[KEY.TRAIN_DENOMINTAOR]

    use_bias_in_linear = config[KEY.USE_BIAS_IN_LINEAR]

    _normalize_sph = config[KEY._NORMALIZE_SPH]
    # model definitions
    edge_embedding = EdgeEmbedding(
        # operate on ||r||
        basis_module=radial_basis_module,
        cutoff_module=cutoff_function_module,
        # operate on r/||r||
        spherical_module=SphericalEncoding(lmax_edge, -1 if is_parity else 1, normalize=_normalize_sph),
    )
    if not parallel:
        layers.update({
            # simple edge preprocessor module with no param
            'edge_preprocess': EdgePreprocess(is_stress=True),
        })

    layers.update({
        # 'Not' simple edge embedding module
        'edge_embedding': edge_embedding,
    })
    # ~~ node embedding to first irreps feature ~~ #
    # here, one hot embedding is preprocess of data not part of model
    # see AtomGraphData._data_for_E3_equivariant_model

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
    if parallel:
        layers.update({
            # Do not change its name (or see deploy.py before change)
            'one_hot_ghost': OnehotEmbedding(
                data_key_x=KEY.NODE_FEATURE_GHOST,
                num_classes=num_species,
                data_key_save=None,
                data_key_additional=None,
            ),
            # Do not change its name (or see deploy.py before change)
            'ghost_onehot_to_feature_x': IrrepsLinear(
                irreps_in=one_hot_irreps,
                irreps_out=irreps_x,
                data_key_in=KEY.NODE_FEATURE_GHOST,
                biases=use_bias_in_linear,
            ),
        })

    # ~~ edge feature(convoluiton filter) ~~ #

    # here, we can infer irreps or weight of tp from lmax and f0_irreps
    # get all possible irreps from tp (here we drop l > lmax)
    irreps_node_attr = one_hot_irreps

    weight_nn_hidden = config[KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS]
    # output layer determined at each IrrepsConvolution layer
    weight_nn_layers = [radial_basis_num] + weight_nn_hidden

    for i in range(num_convolution_layer):
        # here, we can infer irreps of x after interaction from lmax and f0_irreps
        interaction_block = {}

        parity_mode = "full"
        if i == num_convolution_layer - 1:
            lmax_node = 0
            parity_mode = "even"

        # raw irreps out after message(convolution) function
        tp_irreps_out = util.infer_irreps_out(
            irreps_x,  # node feature irreps
            irreps_spherical_harm,  # filter irreps
            drop_l=lmax_node,
            parity_mode=parity_mode,
        )

        # multiplicity maintained irreps after Gate, linear, ..
        true_irreps_out = (
            util.infer_irreps_out(
                irreps_x,
                irreps_spherical_harm,
                drop_l=lmax_node,
                parity_mode=parity_mode,
                fix_multiplicity=feature_multiplicity,
            )
            if irreps_manual is None
            else irreps_manual[i + 1]
        )

        # output irreps of linear 2 & self_connection is determined by Gate
        # Gate require extra scalars(or weight) for l>0 features in nequip,
        # they make it from linear2. (and self_connection have to fit its dimension)
        # Here, initialize gate first and put it later
        gate_layer = EquivariantGate(true_irreps_out, act_scalar, act_gate)
        irreps_for_gate_in = gate_layer.get_gate_irreps_in()

        # from here, data flow split into self connection part and convolution part
        # self connection part is represented as Intro, Outro pair

        # note that this layer does not overwrite x, it calculates tp of in & operand
        # and save its results in somewhere to concatenate to new_x at Outro

        interaction_block[f'{i}_self_connection_intro'] = sc_intro(
            irreps_x=irreps_x,
            irreps_operand=irreps_node_attr,
            irreps_out=irreps_for_gate_in,
        )

        interaction_block[f'{i}_self_interaction_1'] = IrrepsLinear(
            irreps_x,
            irreps_x,
            data_key_in=KEY.NODE_FEATURE,
            biases=use_bias_in_linear,
        )

        if parallel and i == 0:
            # Do not change its name (or see deploy.py before change)
            interaction_block[f'ghost_{i}_self_interaction_1'] = IrrepsLinear(
                irreps_x,
                irreps_x,
                data_key_in=KEY.NODE_FEATURE_GHOST,
                biases=use_bias_in_linear,
            )
        elif parallel and i != 0:
            layers_idx += 1
            layers.update(interaction_block)
            interaction_block = {}  # TODO: this is confusing
            #######################################################
            radial_basis_module, _ = init_radial_basis(config)
            cutoff_function_module = init_cutoff_function(config)
            interaction_block.update({
                'edge_embedding': EdgeEmbedding(
                    # operate on ||r||
                    basis_module=radial_basis_module,
                    cutoff_module=cutoff_function_module,
                    # operate on r/||r||
                    spherical_module=SphericalEncoding(lmax_edge, -1 if is_parity else 1, normalize=_normalize_sph),
                )
            })
            #######################################################
            layers = layers_list[layers_idx]
            # communication from lammps here

        # convolution part, l>lmax is droped as defined in irreps_out
        interaction_block[f'{i}_convolution'] = IrrepsConvolution(
            irreps_x=irreps_x,
            irreps_filter=irreps_spherical_harm,
            irreps_out=tp_irreps_out,
            data_key_weight_input=data_key_weight_input,
            weight_layer_input_to_hidden=weight_nn_layers,
            weight_layer_act=act_radial,
            # TODO: BOTNet says no sqrt is better
            denominator=conv_denominator[i],
            train_denominator=train_conv_denominator,
            is_parallel=parallel,
        )

        # irreps of x increase to gate_irreps_in
        interaction_block[f'{i}_self_interaction_2'] = IrrepsLinear(
            tp_irreps_out,
            irreps_for_gate_in,
            data_key_in=KEY.NODE_FEATURE,
            biases=use_bias_in_linear,
        )

        interaction_block[f'{i}_self_connection_outro'] = sc_outro()

        # irreps of x change back to 'irreps_out'
        interaction_block[f'{i}_equivariant_gate'] = gate_layer
        # now we have irreps of x as 'irreps_out' as we wanted

        layers.update(interaction_block)
        irreps_x = true_irreps_out

        # end of interaction block for-loop

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
    if not parallel:
        fso = ForceStressOutput(
            data_key_energy=KEY.PRED_TOTAL_ENERGY,
            data_key_force=KEY.PRED_FORCE,
            data_key_stress=KEY.PRED_STRESS,
        )
        fof = ForceOutputFromEdge(
            data_key_energy=KEY.PRED_TOTAL_ENERGY,
            data_key_force=KEY.PRED_FORCE,
        )
        gradient_module = fso if not parallel else fof
        layers.update({'force_output': gradient_module})

    # output extraction part
    type_map = config[KEY.TYPE_MAP]
    if parallel:
        return [AtomGraphSequential(v, cutoff, type_map) for v in layers_list]
    else:
        return AtomGraphSequential(layers, cutoff, type_map)
