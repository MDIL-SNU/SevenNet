import os

import torch

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.sevenn_logger import Logger
from sevenn.train.dataset import AtomGraphDataset
from sevenn.train.dataload import parse_structure_list


def init_dataset_from_structure_list(data_config, is_stress, inference=False):
    cutoff = data_config[KEY.CUTOFF]
    # chemical_species = data_config[KEY.CHEMICAL_SPECIES]
    format_outputs = data_config[KEY.FORMAT_OUTPUTS]
    structure_list_files = data_config[KEY.STRUCTURE_LIST]
    model_type = data_config[KEY.MODEL_TYPE]

    type_map = data_config[KEY.TYPE_MAP]

    if model_type == 'E3_equivariant_model':

        #TODO: remove is_stress and put requires grad somewhere adequate
        def preprocessor(x):
            if inference:  # This is only for debugging
                return AtomGraphData.poscar_for_E3_equivariant_model(x, cutoff, type_map, is_stress)
            else:
                return AtomGraphData.data_for_E3_equivariant_model(x, cutoff, type_map, is_stress)
    elif model_type == 'new awesome model':
        pass
    else:
        raise ValueError('unknown model type')

    if type(structure_list_files) is str:
        structure_list_files = [structure_list_files]

    full_dataset = None
    for structure_list in structure_list_files:
        Logger().write(f"loading {structure_list} (it takes several minitues..)\n")
        raw_dct = parse_structure_list(structure_list, format_outputs)
        dataset = AtomGraphDataset(raw_dct, preprocessor, metadata=data_config)
        if full_dataset is None:
            full_dataset = dataset
        else:
            full_dataset.augment(dataset)
        Logger().write(f"loading {structure_list} is done\n")
        # Logger().write(f"current dataset size is :{full_dataset.len()}\n")
        Logger().format_k_v("\ncurrent dataset size is",
                            full_dataset.len(), write=True)

    return full_dataset


def init_dataset(data_config, working_dir, is_stress, inference=False):
    full_dataset = None

    if data_config[KEY.STRUCTURE_LIST] is not False:
        Logger().write("Loading dataset from structure lists\n")
        full_dataset = init_dataset_from_structure_list(data_config, is_stress, inference)

    if data_config[KEY.LOAD_DATASET] is not False:
        load_dataset = data_config[KEY.LOAD_DATASET]
        Logger().write("Loading dataset from load_dataset\n")
        if type(load_dataset) is str:
            load_dataset = [load_dataset]
        for dataset_path in load_dataset:
            Logger().write(f"loading {dataset_path}\n")
            if full_dataset is None:
                full_dataset = torch.load(dataset_path)
            else:
                full_dataset.augment(torch.load(dataset_path))
            Logger().write(f"loading {dataset_path} is done\n")
            Logger().format_k_v("current dataset size is",
                                full_dataset.len(), write=True)
            #Logger().write(f"current dataset size is :{full_dataset.len()}\n")

    prefix = f"{os.path.abspath(working_dir)}/"
    save_dataset = data_config[KEY.SAVE_DATASET]
    if save_dataset is not False:
        if save_dataset.endswith('.pt') is False:
            save_dataset += '.pt'
        if (save_dataset.startswith('.') or save_dataset.startswith('/')) is False:
            save_dataset = prefix + save_dataset  # save_data set is plain file name
        full_dataset.save(save_dataset)  # save_dataset contain user define path
        Logger().format_k_v("Dataset saved to", save_dataset, write=True)
        #Logger().write(f"Loaded full dataset saved to : {save_dataset}\n")

    return full_dataset
