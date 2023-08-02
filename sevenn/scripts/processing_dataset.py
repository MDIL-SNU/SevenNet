import os

import torch
from torch_geometric.loader import DataLoader

from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataset import AtomGraphDataset
from sevenn.train.dataload import parse_structure_list, data_for_E3_equivariant_model
from sevenn.scripts.graph_build import label_atoms_dict_to_dataset
from sevenn.sevenn_logger import Logger
import sevenn._keys as KEY


def from_structure_list(data_config):
    cutoff = data_config[KEY.CUTOFF]
    format_outputs = data_config[KEY.FORMAT_OUTPUTS]
    structure_list_files = data_config[KEY.STRUCTURE_LIST]
    model_type = data_config[KEY.MODEL_TYPE]
    ncores = data_config[KEY.PREPROCESS_NUM_CORES]

    if type(structure_list_files) is str:
        structure_list_files = [structure_list_files]

    full_dataset = None
    for structure_list in structure_list_files:
        Logger().write(f"loading {structure_list} (it takes several minitues..)\n")
        Logger().timer_start("parsing structure_list")
        raw_dct = parse_structure_list(structure_list, format_outputs)
        Logger().timer_end("parsing structure_list",
                           f"parsing {structure_list} is done")
        Logger().timer_start("constructing graph")
        dataset = label_atoms_dict_to_dataset(raw_dct, cutoff, ncores, metadata=data_config)
        Logger().timer_end("constructing graph", "constructing graph is done")
        if full_dataset is None:
            full_dataset = dataset
        else:
            full_dataset.augment(dataset)
        Logger().write(f"loading {structure_list} is done\n")
        # Logger().write(f"current dataset size is :{full_dataset.len()}\n")
        Logger().format_k_v("\ncurrent dataset size is",
                            full_dataset.len(), write=True)

    return full_dataset


def init_dataset(data_config, working_dir):
    full_dataset = None

    if data_config[KEY.STRUCTURE_LIST] is not False:
        Logger().write("Loading dataset from structure lists\n")
        full_dataset = from_structure_list(data_config)

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
    save_by_label = data_config[KEY.SAVE_BY_LABEL]
    if save_dataset:
        if save_dataset.endswith('.sevenn_data') is False:
            save_dataset += '.sevenn_data'
        if (save_dataset.startswith('.') or save_dataset.startswith('/')) is False:
            save_dataset = prefix + save_dataset  # save_data set is plain file name
        full_dataset.save(save_dataset)
        Logger().format_k_v("Dataset saved to", save_dataset, write=True)
        #Logger().write(f"Loaded full dataset saved to : {save_dataset}\n")
    if save_by_label:
        full_dataset.save(prefix, True)
        Logger().format_k_v("Dataset saved by label", prefix, write=True)

    return full_dataset


def processing_dataset(config, working_dir):
    # note that type_map is based on user input(chemical_species)
    type_map = config[KEY.TYPE_MAP]

    Logger().write("\nInitializing dataset...\n")
    dataset = init_dataset(config, working_dir)
    Logger().write("Dataset initialization was successful\n")
    is_stress = (config[KEY.IS_TRACE_STRESS] or config[KEY.IS_TRAIN_STRESS])
    if is_stress:
        dataset.toggle_requires_grad_of_data(KEY.POS, True)
    else:
        dataset.toggle_requires_grad_of_data(KEY.EDGE_VEC, True)

    natoms = dataset.get_natoms()
    dataset.x_to_one_hot_idx(type_map)

    Logger().write("\nNumber of atoms in total dataset:\n")
    Logger().natoms_write(natoms)

    # calculate shift and scale from dataset
    ignore_test = not config[KEY.USE_TESTSET]
    train_set, valid_set, test_set = \
        dataset.divide_dataset(config[KEY.RATIO], ignore_test=ignore_test)
    Logger().write("The dataset divided into train, valid, test set\n")
    Logger().write(Logger.format_k_v("training_set size", train_set.len()))
    Logger().write(Logger.format_k_v("validation_set size", valid_set.len()))
    Logger().write(Logger.format_k_v("test_set size", test_set.len()
                                     if test_set is not None else 0))

    Logger().write("\nCalculating shift and scale from training set...\n")
    #shift, scale = train_set.shift_scale_dataset()
    shift = train_set.get_per_atom_energy_mean()
    scale = train_set.get_force_rmse()
    config.update({KEY.SHIFT: shift, KEY.SCALE: scale})
    #valid_set.shift_scale_dataset(shift=shift, scale=scale)
    Logger().write(f"calculated per_atom_energy mean shift is {shift:.6f} eV\n")
    Logger().write(f"calculated force rms scale is {scale:.6f}\n")

    avg_num_neigh = config[KEY.AVG_NUM_NEIGHBOR]
    if avg_num_neigh:
        Logger().write("Calculating average number of neighbor...\n")
        avg_num_neigh = train_set.get_avg_num_neigh()
        Logger().write(f"average number of neighbor is {avg_num_neigh:.6f}\n")
        config[KEY.AVG_NUM_NEIGHBOR] = avg_num_neigh
    else:
        config[KEY.AVG_NUM_NEIGHBOR] = 1

    batch_size = config[KEY.BATCH_SIZE]
    num_workers = config[KEY.NUM_WORKERS]
    # pin_memory = (device != torch.device("cpu"))
    train_loader = DataLoader(train_set.to_list(), batch_size,
                              shuffle=True, num_workers=num_workers,)
    valid_loader = DataLoader(valid_set.to_list(), batch_size,
                              num_workers=num_workers,)
    if test_set is not None:
        #test_set.shift_scale_dataset(shift=shift, scale=scale)
        test_loader = DataLoader(test_set.to_list(), batch_size)

    statistic_values = (avg_num_neigh, shift, scale)
    loaders = (train_loader, valid_loader, test_loader)
    return statistic_values, loaders, dataset.user_labels
