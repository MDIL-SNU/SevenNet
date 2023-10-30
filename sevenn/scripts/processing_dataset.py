import os
import glob
import random

import torch
from torch_geometric.loader import DataLoader

from sevenn.train.dataset import AtomGraphDataset
from sevenn.train.dataload import match_reader, file_to_dataset
from sevenn.util import onehot_to_chem, chemical_species_preprocess
from sevenn.sevenn_logger import Logger
import sevenn._keys as KEY


def dataset_load(file: str, config):
    """
    Wrapping of dataload.file_to_dataset to suppert
    graph prebuilt sevenn_data
    """
    Logger().write(f"Loading {file}\n")
    Logger().timer_start("loading dataset")

    if file.endswith(".sevenn_data"):
        dataset = torch.load(file)
    else:
        reader, _ = match_reader(config[KEY.DATA_FORMAT],
                                 **config[KEY.DATA_FORMAT_ARGS])
        dataset = file_to_dataset(file,
                                  config[KEY.CUTOFF],
                                  config[KEY.PREPROCESS_NUM_CORES],
                                  reader=reader)
    Logger().format_k_v("loaded dataset size is",
                        dataset.len(), write=True)
    Logger().timer_end("loading dataset", "data set loading time")
    return dataset


# TODO: This is toooooooooooooooooo long
def processing_dataset(config, working_dir):
    # note that type_map is based on user input(chemical_species)
    prefix = f"{os.path.abspath(working_dir)}/"
    is_stress = (config[KEY.IS_TRACE_STRESS] or config[KEY.IS_TRAIN_STRESS])

    Logger().write("\nInitializing dataset...\n")
    cutoff = config[KEY.CUTOFF]
    dataset = AtomGraphDataset({}, cutoff)

    load_dataset = config[KEY.LOAD_DATASET]
    if type(load_dataset) is str:
        load_dataset = [load_dataset]

    for file in load_dataset:
        dataset.augment(dataset_load(file, config))
    dataset.group_by_key()  # apply labels inside original datapoint
    dataset.unify_dtypes()  # unify dtypes of all data points
    if is_stress:
        dataset.toggle_requires_grad_of_data(KEY.POS, True)
    else:
        dataset.toggle_requires_grad_of_data(KEY.EDGE_VEC, True)

    if config[KEY.CHEMICAL_SPECIES] == "auto":
        chem_known = dataset.get_species()
        config.update(chemical_species_preprocess(chem_known))

    #--------------- save dataset regardless of train/valid--------------#
    save_dataset = config[KEY.SAVE_DATASET]
    save_by_label = config[KEY.SAVE_BY_LABEL]
    if save_dataset:
        if save_dataset.endswith('.sevenn_data') is False:
            save_dataset += '.sevenn_data'
        if (save_dataset.startswith('.') or save_dataset.startswith('/')) is False:
            save_dataset = prefix + save_dataset  # save_data set is plain file name
        dataset.save(save_dataset)
        Logger().format_k_v("Dataset saved to", save_dataset, write=True)
        #Logger().write(f"Loaded full dataset saved to : {save_dataset}\n")
    if save_by_label:
        dataset.save(prefix, by_label=True)
        Logger().format_k_v("Dataset saved by label", prefix, write=True)
    #--------------------------------------------------------------------#

    # TODO: testset is not used
    ignore_test = not config[KEY.USE_TESTSET]
    if config[KEY.LOAD_VALIDSET]:
        train_set = dataset
        test_set = AtomGraphDataset([], config[KEY.CUTOFF])

        Logger().write("Loading validset from load_validset\n")
        valid_set = AtomGraphDataset({}, cutoff)
        for file in config[KEY.LOAD_VALIDSET]:
            valid_set.augment(dataset_load(file, config))
        valid_set.group_by_key()
        valid_set.unify_dtypes()

        if is_stress:
            valid_set.toggle_requires_grad_of_data(KEY.POS, True)
        else:
            valid_set.toggle_requires_grad_of_data(KEY.EDGE_VEC, True)

        # condition 1: validset chems should be subset of trainset chems
        valid_chems = valid_set.get_species()
        if set(valid_chems).issubset(set(train_set.get_species())) is False:
            raise ValueError("validset chemical species is not subset of trainset")

        # condition 2: validset labels should be subset of trainset labels
        valid_labels = valid_set.user_labels
        train_labels = train_set.user_labels
        if set(valid_labels).issubset(set(train_labels)) is False:
            valid_set = AtomGraphDataset(valid_set.to_list(), cutoff)
            valid_set.rewrite_labels_to_data()
            train_set = AtomGraphDataset(train_set.to_list(), cutoff)
            train_set.rewrite_labels_to_data()
            Logger().write("WARNING! validset labels is not subset of trainset\n")
            Logger().write("We overwrite all the train, valid labels to default.\n")
            Logger().write("Please create validset by sevenn_graph_build with -l\n")

        Logger().write("the validset loaded, load_dataset is now train_set\n")
        Logger().write("the ratio will be ignored\n")
    else:
        train_set, valid_set, test_set = \
            dataset.divide_dataset(config[KEY.RATIO], ignore_test=ignore_test)
        Logger().write(f"The dataset divided into train, valid by {KEY.RATIO}\n")

    Logger().format_k_v("\nloaded trainset size is", train_set.len(), write=True)
    Logger().format_k_v("\nloaded validset size is", valid_set.len(), write=True)

    Logger().write("Dataset initialization was successful\n")

    Logger().write("\nNumber of atoms in the train_set:\n")
    Logger().natoms_write(train_set.get_natoms(config[KEY.TYPE_MAP]))

    Logger().bar()
    Logger().write("Per atom energy(eV/atom) distribution:\n")
    Logger().statistic_write(train_set.get_statistics(KEY.PER_ATOM_ENERGY))
    Logger().bar()
    Logger().write("Force(eV/Angstrom) distribution:\n")
    Logger().statistic_write(train_set.get_statistics(KEY.FORCE))
    Logger().bar()
    Logger().write("Stress(eV/Angstrom^3) distribution:\n")
    try:
        Logger().statistic_write(train_set.get_statistics(KEY.STRESS))
    except KeyError:
        Logger().write("\n Stress is not included in the train_set\n")
        if is_stress:
            is_stress = False
            Logger().write("Turn off stress training\n")
    Logger().bar()

    # If I did right job, saved data must have atomic numbers as X not one hot idx
    if config[KEY.SAVE_BY_TRAIN_VALID]:
        train_set.save(prefix + "train")
        valid_set.save(prefix + "valid")
        Logger().format_k_v("Dataset saved by train, valid", prefix, write=True)

    # TODO: Why it was needed..?
    #_, _ = train_set.seperate_info()
    #_, _ = valid_set.seperate_info()

    # make sure x is one hot index
    if train_set.x_is_one_hot_idx is False:
        train_set.x_to_one_hot_idx(config[KEY.TYPE_MAP])
    if valid_set.x_is_one_hot_idx is False:
        valid_set.x_to_one_hot_idx(config[KEY.TYPE_MAP])

    Logger().write(Logger.format_k_v("training_set size", train_set.len()))
    Logger().write(Logger.format_k_v("validation_set size", valid_set.len()))

    Logger().write("\nCalculating shift and scale from training set...\n")
    if not config[KEY.USE_SPECIES_WISE_SHIFT_SCALE]:
        shift = train_set.get_per_atom_energy_mean()
        scale = train_set.get_force_rms()
        Logger().write(f"calculated per_atom_energy mean shift is {shift:.6f} eV\n")
        Logger().write(f"calculated force rms scale is {scale:.6f} eV/Angstrom\n")
    else:
        type_map = config[KEY.TYPE_MAP]
        n_chem = len(type_map)
        shift = \
            train_set.get_species_ref_energy_by_linear_comb(n_chem)
        scale = \
            train_set.get_species_wise_force_rms()
        chem_strs = onehot_to_chem(list(range(n_chem)), type_map)
        Logger().write("Calculated specie wise shift, scale is\n")
        for k, (sft, scl) in zip(chem_strs, zip(shift, scale)):
            Logger().write(f"{k:<3} : {sft:.6f} eV, {scl:.6f} eV/Angstrom\n")

    if config[KEY.SHIFT] is not False:
        shift = config[KEY.SHIFT]
        if type(shift) != list and config[KEY.USE_SPECIES_WISE_SHIFT_SCALE]:
            shift = [shift] * len(type_map)
        Logger().write(f"User defined shift found: overwrite shift to {shift}\n")
    if config[KEY.SCALE] is not False:
        scale = config[KEY.SCALE]
        if type(scale) != list and config[KEY.USE_SPECIES_WISE_SHIFT_SCALE]:
            scale = [scale] * len(type_map)
        Logger().write(f"User defined scale found: overwrite scale to {scale}\n")

    config.update({KEY.SHIFT: shift, KEY.SCALE: scale})

    if config[KEY.AVG_NUM_NEIGHBOR] is not False:
        Logger().write("Calculating average number of neighbor...\n")
        avg_num_neigh = train_set.get_avg_num_neigh()
        Logger().write(f"average number of neighbor is {avg_num_neigh:.6f}\n")
        config[KEY.AVG_NUM_NEIGHBOR] = avg_num_neigh
    else:
        config[KEY.AVG_NUM_NEIGHBOR] = 1

    batch_size = config[KEY.BATCH_SIZE]
    num_workers = config[KEY.NUM_WORKERS]

    data_lists = (train_set.to_list(), valid_set.to_list(), test_set.to_list())

    if config[KEY.DATA_SHUFFLE]:
        Logger().write("Shuffle the train data\n")
        for data_list in data_lists:
            random.shuffle(data_list)

    return data_lists
