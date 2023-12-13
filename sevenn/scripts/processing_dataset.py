import os
import glob
import random

import torch

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


def handle_shift_scale(config, train_set, checkpoint_given):
    shift, scale, avg_num_neigh = None, None, None

    Logger().writeline("\nCalculating statistic values from dataset")
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
        Logger().write("calculated specie wise shift, scale is\n")
        for cstr, sh, sc in zip(chem_strs, shift, scale):
            Logger().format_k_v(f"{cstr}", f"{sh:.6f}, {sc:.6f}", write=True)
    avg_num_neigh = train_set.get_avg_num_neigh()
    Logger().format_k_v("avg_num_neigh", f"{avg_num_neigh:.6f}", write=True)

    if checkpoint_given \
            and config[KEY.CONTINUE][KEY.USE_STATISTIC_VALUES_OF_CHECKPOINT]:
        Logger().writeline("Overwrite shift, scale, avg_num_neigh from checkpoint")
        # Values extracted from checkpoint in processing_continue.py
        shift = config[KEY.SHIFT + "_cp"]
        scale = config[KEY.SCALE + "_cp"]
        avg_num_neigh = config[KEY.AVG_NUM_NEIGH + "_cp"]

    # overwrite shift scale anyway if defined in yaml.
    if config[KEY.SHIFT] is not False:
        Logger().writeline("Overwrite shift to value given in yaml")
        if type(config[KEY.SHIFT]) is float \
                and config[KEY.USE_SPECIES_WISE_SHIFT_SCALE]:
            shift = [config[KEY.SHIFT]] * len(config[KEY.TYPE_MAP])
        else:
            shift = config[KEY.SHIFT]
    if config[KEY.SCALE] is not False:
        Logger().writeline("Overwrite scale to value given in yaml")
        if type(config[KEY.SCALE]) is float \
                and config[KEY.USE_SPECIES_WISE_SHIFT_SCALE]:
            scale = [config[KEY.SCALE]] * len(config[KEY.TYPE_MAP])
        else:
            scale = config[KEY.SCALE]
    if type(config[KEY.AVG_NUM_NEIGH]) in [float, list]:
        Logger().writeline("Overwrite avg_num_neigh to value given in yaml")
        avg_num_neigh = config[KEY.AVG_NUM_NEIGH]

    if type(avg_num_neigh) is float:
        avg_num_neigh = [avg_num_neigh] * config[KEY.NUM_CONVOLUTION]

    return shift, scale, avg_num_neigh


# TODO: This is too long
def processing_dataset(config, working_dir):
    prefix = f"{os.path.abspath(working_dir)}/"
    is_stress = (config[KEY.IS_TRACE_STRESS] or config[KEY.IS_TRAIN_STRESS])
    checkpoint_given = config[KEY.CONTINUE][KEY.CHECKPOINT] is not False
    cutoff = config[KEY.CUTOFF]

    Logger().write("\nInitializing dataset...\n")

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

    chem_in_db = dataset.get_species()
    if config[KEY.CHEMICAL_SPECIES] == "auto" and not checkpoint_given:
        config.update(chemical_species_preprocess(chem_in_db))

    # basic dataset compatibility check with previous model
    if checkpoint_given:
        chem_from_cp = config[KEY.CHEMICAL_SPECIES]
        if not all(chem in chem_from_cp for chem in chem_in_db):
            raise ValueError("Chemical species in checkpoint is not compatible")

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

    # inconsistent .info dict give error when collate
    _, _ = train_set.seperate_info()
    _, _ = valid_set.seperate_info()

    # make sure x is one hot index
    if train_set.x_is_one_hot_idx is False:
        train_set.x_to_one_hot_idx(config[KEY.TYPE_MAP])
    if valid_set.x_is_one_hot_idx is False:
        valid_set.x_to_one_hot_idx(config[KEY.TYPE_MAP])

    Logger().write(Logger.format_k_v("training_set size", train_set.len()))
    Logger().write(Logger.format_k_v("validation_set size", valid_set.len()))

    shift, scale, avg_num_neigh =\
        handle_shift_scale(config, train_set, checkpoint_given)
    config.update({KEY.SHIFT: shift,
                   KEY.SCALE: scale,
                   KEY.AVG_NUM_NEIGH: avg_num_neigh})

    data_lists = (train_set.to_list(), valid_set.to_list(), test_set.to_list())
    if config[KEY.DATA_SHUFFLE]:
        Logger().write("Shuffle the train data\n")
        for data_list in data_lists:
            random.shuffle(data_list)

    return data_lists
