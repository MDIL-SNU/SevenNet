from typing import Union
from collections import OrderedDict
from copy import deepcopy

import e3nn.o3
from e3nn.o3 import Irreps
from e3nn.o3 import FullTensorProduct
from torch.nn import Sequential

from sevenn.nn.node_embedding import OnehotEmbedding
from sevenn.nn.edge_embedding import EdgeEmbedding, EdgePreprocess,\
    PolynomialCutoff, XPLORCutoff, BesselBasis, SphericalEncoding
from sevenn.nn.force_output import ForceOutput, ForceOutputFromEdge, \
    ForceOutputFromEdgeParallel, ForceStressOutput
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.nn.linear import IrrepsLinear, AtomReduce, FCN_e3nn
from sevenn.nn.self_connection import SelfConnectionIntro, SelfConnectionOutro
from sevenn.nn.convolution import IrrepsConvolution
from sevenn.nn.equivariant_gate import EquivariantGate
from sevenn.nn.activation import ShiftedSoftPlus
from sevenn.nn.scale import Rescale, SpeciesWiseRescale

import sevenn._keys as KEY
import sevenn._const as _const


def infer_irreps_out(irreps_x: Irreps,
                     irreps_operand: Irreps,
                     drop_l: Union[bool, int] = False,
                     fix_multiplicity: Union[bool, int] = False,
                     only_even_p: bool = False):
    # (mul, (ir, p))
    irreps_out = FullTensorProduct(irreps_x, irreps_operand).irreps_out.simplify()
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
        return XPLORCutoff(cutoff, cutoff_function_dct["cutoff_on"])

    raise RuntimeError('something went very wrong...')


def build_E3_equivariant_model(model_config: dict, parallel=False):
    """
    IDENTICAL to nequip model
    atom embedding is not part of model
    """
    feature_multiplicity = model_config[KEY.NODE_FEATURE_MULTIPLICITY]
    lmax = model_config[KEY.LMAX]
    lmax_edge = model_config[KEY.LMAX_EDGE] \
        if model_config[KEY.LMAX_EDGE] >= 0 else lmax
    lmax_node = model_config[KEY.LMAX_NODE] \
        if model_config[KEY.LMAX_NODE] >= 0 else lmax
    num_convolution_layer = model_config[KEY.NUM_CONVOLUTION]
    is_parity = model_config[KEY.IS_PARITY]  # boolean
    is_stress = \
        model_config[KEY.IS_TRACE_STRESS] or model_config[KEY.IS_TRAIN_STRESS]
    num_species = model_config[KEY.NUM_SPECIES]
    irreps_spherical_harm =\
        Irreps.spherical_harmonics(lmax_edge, -1 if is_parity else 1)
    if parallel:
        layers_list = [OrderedDict() for _ in range(num_convolution_layer)]
        layers_idx = 0
        layers = layers_list[0]
    else:
        layers = OrderedDict()

    irreps_manual = None
    if model_config[KEY.IRREPS_MANUAL] is not False:
        irreps_manual = model_config[KEY.IRREPS_MANUAL]
        try:
            irreps_manual = [Irreps(irr) for irr in irreps_manual]
            assert len(irreps_manual) == num_convolution_layer + 1
        except Exception:
            raise RuntimeError('invalid irreps_manual input given')

    optimize_by_reduce = model_config[KEY.OPTIMIZE_BY_REDUCE]
    use_bias_in_linear = model_config[KEY.USE_BIAS_IN_LINEAR]

    act_scalar = {}
    act_gate = {}
    for (k, v) in model_config[KEY.ACTIVATION_SCARLAR].items():
        act_scalar[k] = _const.ACTIVATION_DICT[k][v]
    for (k, v) in model_config[KEY.ACTIVATION_GATE].items():
        act_gate[k] = _const.ACTIVATION_DICT[k][v]
    act_radial = _const.ACTIVATION[model_config[KEY.ACTIVATION_RADIAL]]

    # ~~ edge embedding ~~ #
    cutoff = model_config[KEY.CUTOFF]

    radial_basis_module, radial_basis_num = init_radial_basis(model_config)
    cutoff_function_module = init_cutoff_function(model_config)

    avg_num_neigh = model_config[KEY.AVG_NUM_NEIGH]
    if type(avg_num_neigh) is not list:
        avg_num_neigh = [avg_num_neigh] * num_convolution_layer
    train_avg_num_neigh = model_config[KEY.TRAIN_AVG_NUM_NEIGH]

    edge_embedding = EdgeEmbedding(
        # operate on ||r||
        basis_module=radial_basis_module,
        cutoff_module=cutoff_function_module,
        # operate on r/||r||
        spherical_module=SphericalEncoding(lmax_edge, -1 if is_parity else 1),
    )
    if is_stress:
        layers.update(
            {
                # simple edge preprocessor module with no param
                "EdgePreprocess": EdgePreprocess(is_stress),
            }
        )

    layers.update(
        {
            # 'Not' simple edge embedding module
            "EdgeEmbedding": edge_embedding,
        }
    )
    # ~~ node embedding to first irreps feature ~~ #
    # here, one hot embedding is preprocess of data not part of model
    # see AtomGraphData._data_for_E3_equivariant_model

    one_hot_irreps = Irreps(f"{num_species}x0e")
    irreps_x = Irreps(f"{feature_multiplicity}x0e") if irreps_manual is None \
        else irreps_manual[0]
    layers.update(
        {
            "onehot_idx_to_onehot":
            OnehotEmbedding(
                num_classes=num_species
            ),
            "onehot_to_feature_x":
            IrrepsLinear(
                irreps_in=one_hot_irreps,
                irreps_out=irreps_x,
                data_key_in=KEY.NODE_FEATURE,
                biases=use_bias_in_linear
            )
        }
    )
    if parallel:
        layers.update(
            {
                "one_hot_ghost":
                OnehotEmbedding(
                    data_key_x=KEY.NODE_FEATURE_GHOST,
                    num_classes=num_species,
                    data_key_save=None,
                    data_key_additional=None
                ),
                "ghost_onehot_to_feature_x":
                IrrepsLinear(
                    irreps_in=one_hot_irreps,
                    irreps_out=irreps_x,
                    data_key_in=KEY.NODE_FEATURE_GHOST,
                    biases=use_bias_in_linear
                )
            }
        )

    # ~~ edge feature(convoluiton filter) ~~ #

    # here, we can infer irreps or weight of tp from lmax and f0_irreps
    # get all possible irreps from tp (here we drop l > lmax)
    irreps_node_attr = one_hot_irreps

    weight_nn_hidden = model_config[KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS]
    # output layer determined at each IrrepsConvolution layer
    weight_nn_layers = [radial_basis_num] + weight_nn_hidden

    for i in range(num_convolution_layer):
        # here, we can infer irreps of x after interaction from lmax and f0_irreps
        interaction_block = {}

        only_even_p = False
        if optimize_by_reduce:
            if i == num_convolution_layer - 1:
                lmax_node = 0
                only_even_p = True

        # raw irreps out after message(convolution) function
        tp_irreps_out = infer_irreps_out(irreps_x,  # node feature irreps
                                         irreps_spherical_harm,  # filter irreps
                                         drop_l=lmax_node,
                                         only_even_p=only_even_p)

        # multiplicity maintained irreps after Gate, linear, ..
        true_irreps_out = infer_irreps_out(irreps_x,
                                           irreps_spherical_harm,
                                           drop_l=lmax_node,
                                           fix_multiplicity=feature_multiplicity,
                                           only_even_p=only_even_p) if \
                          irreps_manual is None else \
                          irreps_manual[i + 1]

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

        interaction_block[f"{i} self connection intro"] = \
            SelfConnectionIntro(irreps_x=irreps_x,
                                irreps_operand=irreps_node_attr,
                                irreps_out=irreps_for_gate_in)

        interaction_block[f"{i}_self_interaction_1"] = \
            IrrepsLinear(irreps_x,
                         irreps_x,
                         data_key_in=KEY.NODE_FEATURE,
                         biases=use_bias_in_linear)

        if parallel and i == 0:
            interaction_block[f"ghost_{i}_self_interaction_1"] = \
                IrrepsLinear(irreps_x,
                             irreps_x,
                             data_key_in=KEY.NODE_FEATURE_GHOST,
                             biases=use_bias_in_linear)
        elif parallel and i != 0:
            layers_idx += 1
            layers.update(interaction_block)
            interaction_block = {}  # TODO: this is confusing
            #######################################################
            radial_basis_module, _ = init_radial_basis(model_config)
            cutoff_function_module = init_cutoff_function(model_config)
            interaction_block.update(
                {
                    "EdgeEmbedding": EdgeEmbedding(
                        # operate on ||r||
                        basis_module=radial_basis_module,
                        cutoff_module=cutoff_function_module,
                        # operate on r/||r||
                        spherical_module=SphericalEncoding(lmax_edge),
                    )
                }
            )
            #######################################################
            layers = layers_list[layers_idx]
            # communication from lammps here

        # convolution part, l>lmax is droped as defined in irreps_out
        interaction_block[f"{i} convolution"] = \
            IrrepsConvolution(
                irreps_x=irreps_x,
                irreps_filter=irreps_spherical_harm,
                irreps_out=tp_irreps_out,
                weight_layer_input_to_hidden=weight_nn_layers,
                weight_layer_act=act_radial,
                # TODO: BOTNet says no sqrt is better
                denumerator=avg_num_neigh[i]**0.5,
                train_denumerator=train_avg_num_neigh,
                is_parallel=parallel)

        # irreps of x increase to gate_irreps_in
        interaction_block[f"{i} self interaction 2"] = \
            IrrepsLinear(tp_irreps_out,
                         irreps_for_gate_in,
                         data_key_in=KEY.NODE_FEATURE,
                         biases=use_bias_in_linear)

        interaction_block[f"{i} self connection outro"] = \
            SelfConnectionOutro()

        # irreps of x change back to 'irreps_out'
        interaction_block[f"{i} equivariant gate"] = gate_layer
        # now we have irreps of x as 'irreps_out' as we wanted

        layers.update(interaction_block)
        irreps_x = true_irreps_out

        # end of interaction block for-loop

    if model_config[KEY.READOUT_AS_FCN] is False:
        mid_dim = feature_multiplicity if irreps_manual is None else \
            irreps_manual[-1].num_irreps
        hidden_irreps = Irreps([(mid_dim // 2, (0, 1))])
        layers.update(
            {
                "reducing nn input to hidden":
                IrrepsLinear(
                    irreps_x,
                    hidden_irreps,
                    data_key_in=KEY.NODE_FEATURE,
                    biases=use_bias_in_linear
                ),
                "reducing nn hidden to energy":
                IrrepsLinear(
                    hidden_irreps,
                    Irreps([(1, (0, 1))]),
                    data_key_in=KEY.NODE_FEATURE,
                    data_key_out=KEY.SCALED_ATOMIC_ENERGY,
                    biases=use_bias_in_linear
                ),
            }
        )
    else:
        act =\
            _const.ACTIVATION[model_config[KEY.READOUT_FCN_ACTIVATION]]
        hidden_neurons = model_config[KEY.READOUT_FCN_HIDDEN_NEURONS]
        layers.update(
            {
                "readout_FCN":
                FCN_e3nn(
                    dim_out=1,
                    hidden_neurons=hidden_neurons,
                    activation=act,
                    data_key_in=KEY.NODE_FEATURE,
                    data_key_out=KEY.SCALED_ATOMIC_ENERGY,
                    irreps_in=irreps_x,
                )
            }
        )

    shift = model_config[KEY.SHIFT]
    scale = model_config[KEY.SCALE]
    train_shift_scale = model_config[KEY.TRAIN_SHIFT_SCALE]
    rescale_module = SpeciesWiseRescale \
        if model_config[KEY.USE_SPECIES_WISE_SHIFT_SCALE] \
        else Rescale
    layers.update(
        {
            "rescale atomic energy":
            rescale_module(
                shift=shift,
                scale=scale,
                data_key_in=KEY.SCALED_ATOMIC_ENERGY,
                data_key_out=KEY.ATOMIC_ENERGY,
                train_shift_scale=train_shift_scale,
            ),
            "reduce to total enegy":
            AtomReduce(
                data_key_in=KEY.ATOMIC_ENERGY,
                data_key_out=KEY.PRED_TOTAL_ENERGY,
                constant=1.0,
            ),
        }
    )
    if not parallel:
        fso = ForceStressOutput(data_key_energy=KEY.PRED_TOTAL_ENERGY,
                                data_key_force=KEY.PRED_FORCE,
                                data_key_stress=KEY.PRED_STRESS)
        fof = ForceOutputFromEdge(data_key_energy=KEY.PRED_TOTAL_ENERGY,
                                  data_key_force=KEY.PRED_FORCE)
        gradient_module = fso if is_stress else fof
        layers.update({"force output": gradient_module})

    # output extraction part
    if parallel:
        return [AtomGraphSequential(v) for v in layers_list]
    else:
        return AtomGraphSequential(layers)
