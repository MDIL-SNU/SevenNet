from typing import Union
from collections import OrderedDict
from copy import deepcopy

import e3nn.o3
from e3nn.o3 import Irreps
from e3nn.o3 import FullTensorProduct
from torch.nn import Sequential

import sevenn.nn.node_embedding
from sevenn.nn.ghost_control import GhostControlCat, GhostControlSplit
from sevenn.nn.edge_embedding import EdgeEmbedding, EdgePreprocess,\
    PolynomialCutoff, BesselBasis, SphericalEncoding
from sevenn.nn.force_output import ForceOutput, ForceOutputFromEdge, \
    ForceOutputFromEdgeParallel
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.nn.linear import IrrepsLinear, AtomReduce
from sevenn.nn.self_connection import SelfConnectionIntro, SelfConnectionOutro
from sevenn.nn.convolution import IrrepsConvolution
from sevenn.nn.equivariant_gate import EquivariantGate
from sevenn.nn.activation import ShiftedSoftPlus
from sevenn.nn.scale import Scale
from sevenn.nn.grads_calc import GradsCalc

import sevenn._keys as KEY
import sevenn._const as _const


def infer_irreps_out(irreps_x: Irreps,
                     irreps_operand: Irreps,
                     drop_l: Union[bool, int] = False,
                     fix_multiplicity: Union[bool, int] = False):
    # (mul, (ir, p))
    irreps_out = FullTensorProduct(irreps_x, irreps_operand).irreps_out.simplify()
    new_irreps_elem = []
    for mul, (l, p) in irreps_out:
        elem = (mul, (l, p))
        if drop_l is not False and l > drop_l:
            continue
        if fix_multiplicity is not False:
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


def init_cutoff_function(config):
    cutoff_function_dct = config[KEY.CUTOFF_FUNCTION]
    cutoff = config[KEY.CUTOFF]

    if cutoff_function_dct[KEY.CUTOFF_FUNCTION_NAME] == 'poly_cut':
        p = cutoff_function_dct[KEY.POLY_CUT_P]
        return PolynomialCutoff(p, cutoff)

    raise RuntimeError('something went very wrong...')


def build_E3_equivariant_model(model_config: dict, parallel=False):
    """
    identical to nequip model
    atom embedding is not part of model (its input preprocessing)
    No ResNet style update (but self connection yes)

    parallel here is bad considering code readability & maintanence
    appropriate place for logic for parallel is deploy_parallel()
    but inserting extra layers & splitting model is hard after the
    model buliding. So code remains here.
    """
    feature_multiplicity = model_config[KEY.NODE_FEATURE_MULTIPLICITY]
    lmax = model_config[KEY.LMAX]
    num_convolution_layer = model_config[KEY.NUM_CONVOLUTION]
    is_parity = model_config[KEY.IS_PARITY]  # boolean
    num_species = model_config[KEY.NUM_SPECIES]
    irreps_spherical_harm = Irreps.spherical_harmonics(lmax, -1 if is_parity else 1)
    if parallel:
        layers_list = [OrderedDict() for _ in range(num_convolution_layer)]
        layers_idx = 0
        layers = layers_list[0]
    else:
        layers = OrderedDict()

    shift = model_config[KEY.SHIFT]
    scale = model_config[KEY.SCALE]

    act_gate = {}
    act_scalar = {}
    for (k, v) in model_config[KEY.ACTIVATION_SCARLAR].items():
        act_gate[k] = _const.ACTIVATION_DICT[k][v]
    for (k, v) in model_config[KEY.ACTIVATION_GATE].items():
        act_scalar[k] = _const.ACTIVATION_DICT[k][v]

    # ~~ edge embedding ~~ #
    cutoff = model_config[KEY.CUTOFF]

    radial_basis_module, radial_basis_num = init_radial_basis(model_config)
    cutoff_function_module = init_cutoff_function(model_config)

    avg_num_neigh = model_config[KEY.AVG_NUM_NEIGHBOR]

    edge_embedding = EdgeEmbedding(
        # operate on ||r||
        basis_module=radial_basis_module,
        cutoff_module=cutoff_function_module,
        # operate on r/||r||
        spherical_module=SphericalEncoding(lmax),
    )
    layers.update(
        {
            # simple edge preprocessor module with no param
            #"EdgePreprocess": EdgePreprocess(),
            # 'Not' simple edge embedding module
            "EdgeEbmedding": edge_embedding,
        }
    )
    # ~~ node embedding to first irreps feature ~~ #
    # here, one hot embedding is preprocess of data not part of model
    # see AtomGraphData._data_for_E3_equivariant_model

    one_hot_irreps = Irreps(f"{num_species}x0e")
    f_in_irreps = Irreps(f"{feature_multiplicity}x0e")
    layers.update(
        {
            "onehot_to_feature_x":
            IrrepsLinear(
                irreps_in=one_hot_irreps,
                irreps_out=f_in_irreps,
                data_key_in=KEY.NODE_FEATURE
            )
        }
    )
    if parallel:
        layers.update(
            {
                "ghost_onehot_to_feature_x":
                IrrepsLinear(
                    irreps_in=one_hot_irreps,
                    irreps_out=f_in_irreps,
                    data_key_in=KEY.NODE_FEATURE_GHOST
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

    # nonlinearity equivariant gate related
    # act_gate = model_config[KEY.ACTIVATION_GATE]
    # act_scalar = model_config[KEY.ACTIVATION_SCARLAR]

    irreps_x = f_in_irreps
    for i in range(num_convolution_layer):
        # here, we can infer irreps of x after interaction from lmax and f0_irreps
        interaction_block = {}

        tp_irreps_out = infer_irreps_out(irreps_x,
                                         irreps_spherical_harm,
                                         drop_l=lmax)

        true_irreps_out = infer_irreps_out(irreps_x,
                                           irreps_spherical_harm,
                                           drop_l=lmax,
                                           fix_multiplicity=feature_multiplicity)

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
            IrrepsLinear(irreps_x, irreps_x, data_key_in=KEY.NODE_FEATURE)

        if parallel and i == 0:
            interaction_block[f"ghost_{i}_self_interaction_1"] = \
                IrrepsLinear(irreps_x, irreps_x, data_key_in=KEY.NODE_FEATURE_GHOST)
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
                        spherical_module=SphericalEncoding(lmax),
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
                weight_layer_act=act_scalar["e"],
                denumerator=avg_num_neigh**0.5,
                is_parallel=parallel)

        # irreps of x increase to gate_irreps_in
        interaction_block[f"{i} self interaction 2"] = \
            IrrepsLinear(tp_irreps_out,
                         irreps_for_gate_in,
                         data_key_in=KEY.NODE_FEATURE)

        interaction_block[f"{i} self connection outro"] = \
            SelfConnectionOutro()

        # irreps of x change back to 'irreps_out'
        interaction_block[f"{i} equivariant gate"] = gate_layer
        # now we have irreps of x as 'irreps_out' as we wanted

        layers.update(interaction_block)
        irreps_x = true_irreps_out

        # end of interaction block for-loop

    hidden_irreps = Irreps([(feature_multiplicity // 2, (0, 1))])

    layers.update(
        {
            "reducing nn input to hidden":
            IrrepsLinear(
                irreps_x,
                hidden_irreps,
                data_key_in=KEY.NODE_FEATURE,
            ),
            "reducing nn hidden to energy":
            IrrepsLinear(
                hidden_irreps,
                Irreps([(1, (0, 1))]),
                data_key_in=KEY.NODE_FEATURE,
                data_key_out=KEY.ATOMIC_ENERGY,
            ),
            "reduce to total enegy":
            AtomReduce(
                data_key_in=KEY.ATOMIC_ENERGY,
                data_key_out=KEY.SCALED_ENERGY,
                #data_key_out=KEY.PRED_TOTAL_ENERGY,
                constant=1.0,
            )
        }
    )
    if not parallel:
        layers.update(
            {
                "force output": ForceOutputFromEdge(
                    #data_key_energy=KEY.PRED_TOTAL_ENERGY,
                    #data_key_force=KEY.PRED_FORCE,
                    data_key_energy=KEY.SCALED_ENERGY,
                    data_key_force=KEY.SCALED_FORCE,
                ),
                "rescale": Scale(
                    shift=shift,
                    scale=scale,
                    scale_per_atom=True,
                )
            }
        )

    # output extraction part
    if parallel:
        return [AtomGraphSequential(v) for v in layers_list]
    else:
        return AtomGraphSequential(layers)


#TODO: move to deploy_parallel
"""
def build_parallel_model(model_ori: AtomGraphSequential, config):
    GHOST_LAYERS_KEYS = ["onehot_to_feature_x", "0_self_interaction_1"]
    num_conv = config[KEY.NUM_CONVOLUTION]

    state_dict_ori = model_ori.state_dict()
    model_list = build_E3_equivariant_model(config, parallel=True)
    dct_temp = {}
    for ghost_layer_key in GHOST_LAYERS_KEYS:
        for key, val in state_dict_ori.items():
            if key.startswith(ghost_layer_key):
                dct_temp.update({f"ghost_{key}": val})
            else:
                continue
    state_dict_ori.update(dct_temp)

    for model_part in model_list:
        model_part.load_state_dict(state_dict_ori, strict=False)
        #stt = model_part.state_dict()
    return model_list
"""


def main():
    import pickle
    import torch
    from atom_graph_data import AtomGraphData

    torch.manual_seed(777)
    config = _const.DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG
    config[KEY.LMAX] = 2
    config[KEY.NUM_CONVOLUTION] = 3
    config[KEY.SHIFT] = 1.0
    config[KEY.SCALE] = 1.0
    model = build_E3_equivariant_model(config)
    #deploy_parallel(model, config, "deployed_test")

    #model_list = build_parallel_model(model, config)
    #print(model_list)

    #model.eval()
    #model.set_is_batch_data(False)
    #stct_dct = model.state_dict()
    #for k in stct_dct.keys():
    #    print(k)

    #model2.load_state_dict(stct_dct, strict=False)


if __name__ == "__main__":
    main()

