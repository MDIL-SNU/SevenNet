import pickle
import copy
from datetime import datetime

import torch
import e3nn.util.jit
from ase.data import chemical_symbols

from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model, build_parallel_model
from sevenn.nn.node_embedding import OnehotEmbedding
from sevenn.nn.sequential import AtomGraphSequential
import sevenn._keys as KEY
import sevenn._const as _const


def deploy_from_compiled(model_ori: AtomGraphSequential, config, fname):
    model_new = build_E3_equivariant_model(config)

    num_species = config[KEY.NUM_SPECIES]
    model_new.prepand_module('one_hot', OnehotEmbedding(num_classes=num_species))
    model_new.set_is_batch_data(False)
    model_new.eval()

    model_new = e3nn.util.jit.script(model_new)
    model_new.load_state_dict(model_ori.state_dict())
    model = torch.jit.freeze(model_new)

    # make some config need for md
    md_configs = {}
    type_map = config[KEY.TYPE_MAP]
    # in lammps, input is chemical symbols so we need chemical_symbol->one_hot_idx
    # type_map = {76: 0, 16: 1}, chem_list = 'Hf O'
    # later in lammps do something like chem_list.split(' ')
    # then its index is one_hot_idx. note that type_map preserve order
    chem_list = ""
    for Z in type_map.keys():
        chem_list += chemical_symbols[Z] + " "
    chem_list.strip()
    md_configs.update({"chemical_symbols_to_index": chem_list})
    md_configs.update({"cutoff": str(config[KEY.CUTOFF])})
    md_configs.update({"num_species": str(config[KEY.NUM_SPECIES])})
    md_configs.update({"model_type": config[KEY.MODEL_TYPE]})
    md_configs.update({"version": _const.SEVENN_VERSION})
    md_configs.update({"dtype": config[KEY.DTYPE]})
    md_configs.update({"time": datetime.now().strftime('%Y-%m-%d')})

    torch.jit.save(model, fname, _extra_files=md_configs)


#TODO: this is E3_equivariant specific
def deploy(model_ori: AtomGraphSequential, config, fname):
    # some postprocess for md mode of model
    model = build_E3_equivariant_model(config)
    model.load_state_dict(model_ori.state_dict())  # copy model

    num_species = config[KEY.NUM_SPECIES]
    model.prepand_module('one_hot', OnehotEmbedding(num_classes=num_species))
    model.set_is_batch_data(False)
    model.eval()

    model = e3nn.util.jit.script(model)
    model = torch.jit.freeze(model)

    # make some config need for md
    md_configs = {}
    type_map = config[KEY.TYPE_MAP]
    # in lammps, input is chemical symbols so we need chemical_symbol->one_hot_idx
    # type_map = {76: 0, 16: 1}, chem_list = 'Hf O'
    # later in lammps do something like chem_list.split(' ')
    # then its index is one_hot_idx. note that type_map preserve order
    chem_list = ""
    for Z in type_map.keys():
        chem_list += chemical_symbols[Z] + " "
    chem_list.strip()
    md_configs.update({"chemical_symbols_to_index": chem_list})
    md_configs.update({"cutoff": str(config[KEY.CUTOFF])})
    md_configs.update({"num_species": str(config[KEY.NUM_SPECIES])})
    md_configs.update({"model_type": config[KEY.MODEL_TYPE]})
    md_configs.update({"version": _const.SEVENN_VERSION})
    md_configs.update({"dtype": config[KEY.DTYPE]})
    md_configs.update({"time": datetime.now().strftime('%Y-%m-%d')})

    torch.jit.save(model, fname, _extra_files=md_configs)

    # validation
    # print(model)


#TODO: this is E3_equivariant specific
def deploy_parallel(model_ori: AtomGraphSequential, config, fname):
    # some postprocess for md mode of model
    model_list = build_parallel_model(model_ori, config)
    if type(model_list) is not list:
        model_list = [model_list]

    num_species = config[KEY.NUM_SPECIES]
    model_list[0].prepand_module('one_hot', OnehotEmbedding(
        data_key_in=KEY.NODE_FEATURE, num_classes=num_species))
    model_list[0].prepand_module('one_hot_ghost', OnehotEmbedding(
        data_key_in=KEY.NODE_FEATURE_GHOST,
        num_classes=num_species,
        data_key_additional=None))
    # make some config need for md
    print(model_list)
    md_configs = {}
    type_map = config[KEY.TYPE_MAP]
    # in lammps, input is chemical symbols so we need chemical_symbol->one_hot_idx
    # type_map = {76: 0, 16: 1}, chem_list = 'Hf O'
    # later in lammps do something like chem_list.split(' ')
    # then its index is one_hot_idx. note that type_map preserve order
    chem_list = ""
    for Z in type_map.keys():
        chem_list += chemical_symbols[Z] + " "
    chem_list.strip()
    md_configs.update({"chemical_symbols_to_index": chem_list})
    md_configs.update({"cutoff": str(config[KEY.CUTOFF])})
    md_configs.update({"num_species": str(config[KEY.NUM_SPECIES])})
    md_configs.update({"shift": str(config[KEY.SHIFT])})
    md_configs.update({"scale": str(config[KEY.SCALE])})
    md_configs.update({"model_type": config[KEY.MODEL_TYPE]})
    md_configs.update({"version": _const.SEVENN_VERSION})
    md_configs.update({"dtype": config[KEY.DTYPE]})
    md_configs.update({"time": datetime.now().strftime('%Y-%m-%d')})

    for idx, model in enumerate(model_list):
        fname_full = f"{fname}_{idx}.pt"
        model.set_is_batch_data(False)
        model.eval()

        model = e3nn.util.jit.script(model)
        model = torch.jit.freeze(model)

        torch.jit.save(model, fname_full, _extra_files=md_configs)

    # validation
    # print(model)


def main():
    from sevenn.nn.node_embedding import get_type_mapper_from_specie
    torch.manual_seed(777)
    config = _const.DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG
    config[KEY.LMAX] = 2
    config[KEY.NUM_CONVOLUTION] = 2
    config[KEY.NODE_FEATURE_MULTIPLICITY] = 8
    config[KEY.SHIFT] = 1.0
    config[KEY.SCALE] = 1.0
    type_map = get_type_mapper_from_specie(['Hf', 'O'])
    config[KEY.TYPE_MAP] = type_map
    config[KEY.CHEMICAL_SPECIES] = ['Hf', 'O']
    config[KEY.NUM_SPECIES] = 2
    config[KEY.MODEL_TYPE] = 'E3_equivariant_model'
    config[KEY.DTYPE] = "single"
    model = build_E3_equivariant_model(config)
    deploy_parallel(model, config, "deployed_test")
    deploy(model, config, "deployed_ref.pt")


if __name__ == "__main__":
    main()

