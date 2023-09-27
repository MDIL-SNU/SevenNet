import os
import glob

import torch
from torch_geometric.loader import DataLoader

from sevenn.util import chemical_species_preprocess
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
        Logger().writeline("constructing graph...")
        dataset = label_atoms_dict_to_dataset(raw_dct, cutoff, ncores)
        dataset.meta = data_config
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


def from_sevenn_data(load_dataset):
    Logger().write("Loading dataset from sevenn_data\n")

    Logger().timer_start("loading dataset")
    dataset = None
    if type(load_dataset) is str:
        load_dataset = [load_dataset]
    for dataset_path in load_dataset:
        files = glob.glob(dataset_path)
        for file in files:
            if not file.endswith(".sevenn_data"):
                continue
            Logger().write(f"loading {file}\n")
            if dataset is None:
                dataset = torch.load(file)
            else:
                dataset.augment(torch.load(file))
            Logger().write(f"loading {file} is done\n")
            Logger().format_k_v("current dataset size from load_dataset is",
                                dataset.len(), write=True)
    Logger().timer_end("loading dataset", "data set loading time")
    return dataset


def init_dataset(data_config, working_dir):
    full_dataset = None

    if data_config[KEY.STRUCTURE_LIST] is not False:
        Logger().write("Loading dataset from structure lists\n")
        full_dataset = from_structure_list(data_config)

    if data_config[KEY.LOAD_DATASET] is not False:
        # TODO: I hate this pattern
        if full_dataset is None:
            full_dataset = from_sevenn_data(data_config[KEY.LOAD_DATASET])
        else:
            full_dataset.augment(from_sevenn_data(data_config[KEY.LOAD_DATASET]))
    Logger().format_k_v("\nfinal dataset size is", full_dataset.len(), write=True)
    full_dataset.group_by_key()  # apply labels inside original datapoint
    full_dataset.unify_dtypes()  # unify dtypes of all data points

    return full_dataset


def processing_dataset(config, working_dir):
    # note that type_map is based on user input(chemical_species)
    prefix = f"{os.path.abspath(working_dir)}/"

    Logger().write("\nInitializing dataset...\n")
    is_stress = (config[KEY.IS_TRACE_STRESS] or config[KEY.IS_TRAIN_STRESS])
    dataset = init_dataset(config, working_dir)
    Logger().write("Dataset initialization was successful\n")

    # initialize type_map chemical_species and so on based on total dataset
    if config[KEY.CHEMICAL_SPECIES] == "auto":
        input_chem = dataset.get_species()
        config.update(chemical_species_preprocess(input_chem))

    Logger().write("\nNumber of atoms in the dataset:\n")
    Logger().natoms_write(dataset.get_natoms(config[KEY.TYPE_MAP]))

    Logger().bar()
    Logger().write("Per atom energy(eV/atom) distribution:\n")
    Logger().statistic_write(dataset.get_statistics(KEY.PER_ATOM_ENERGY))
    Logger().bar()
    Logger().write("Force(eV/Angstrom) distribution:\n")
    Logger().statistic_write(dataset.get_statistics(KEY.FORCE))
    Logger().bar()
    Logger().write("Stress(eV/Angstrom^3) distribution:\n")
    try:
        Logger().statistic_write(dataset.get_statistics(KEY.STRESS))
    except KeyError:
        Logger().write("\n Stress is not included in the dataset\n")
        if is_stress:
            is_stress = False
            Logger().write("Turn off stress training\n")
    Logger().bar()

    if is_stress:
        dataset.toggle_requires_grad_of_data(KEY.POS, True)
    else:
        dataset.delete_data_key(KEY.STRESS)
        dataset.delete_data_key(KEY.CELL)
        dataset.delete_data_key(KEY.CELL_SHIFT)
        dataset.delete_data_key(KEY.CELL_VOLUME)
        dataset.toggle_requires_grad_of_data(KEY.EDGE_VEC, True)

    # calculate shift and scale from dataset
    # TODO: testset is not used
    ignore_test = not config[KEY.USE_TESTSET]
    if config[KEY.LOAD_VALIDSET] is not False:
        train_set = dataset
        test_set = AtomGraphDataset([], config[KEY.CUTOFF])
        Logger().write("Loading validset from load_validset\n")
        valid_set = from_sevenn_data(config[KEY.LOAD_VALIDSET])
        valid_set.group_by_key()
        valid_set.unify_dtypes()
        if is_stress:
            valid_set.toggle_requires_grad_of_data(KEY.POS, True)
        else:
            valid_set.toggle_requires_grad_of_data(KEY.EDGE_VEC, True)
        Logger().write("WARNING! the validset loaded ratio will be ignored\n")
    else:
        train_set, valid_set, test_set = \
            dataset.divide_dataset(config[KEY.RATIO], ignore_test=ignore_test)
        Logger().write(f"The dataset divided into train, valid by {KEY.RATIO}\n")

    # If I did right job, saved data must have atomic numbers as X not one hot idx
    save_dataset = config[KEY.SAVE_DATASET]
    save_by_label = config[KEY.SAVE_BY_LABEL]
    save_by_train_valid = config[KEY.SAVE_BY_TRAIN_VALID]
    if save_dataset:
        if save_dataset.endswith('.sevenn_data') is False:
            save_dataset += '.sevenn_data'
        if (save_dataset.startswith('.') or save_dataset.startswith('/')) is False:
            save_dataset = prefix + save_dataset  # save_data set is plain file name
        dataset.save(save_dataset)
        Logger().format_k_v("Dataset saved to", save_dataset, write=True)
        #Logger().write(f"Loaded full dataset saved to : {save_dataset}\n")
    if save_by_label:
        full_dataset.save(prefix, by_label=True)
        Logger().format_k_v("Dataset saved by label", prefix, write=True)
    if save_by_train_valid:
        train_set.save(prefix + "train")
        valid_set.save(prefix + "valid")
        Logger().format_k_v("Dataset saved by train, valid", prefix, write=True)

    # TODO: Maybe it is need during training for logging?
    #_, _ = train_set.seperate_info()
    #_, _ = valid_set.seperate_info()

    # make sure x is one hot index
    if train_set.x_is_one_hot_idx is False:
        train_set.x_to_one_hot_idx(config[KEY.TYPE_MAP])
    if valid_set.x_is_one_hot_idx is False:
        valid_set.x_to_one_hot_idx(config[KEY.TYPE_MAP])

    Logger().write(Logger.format_k_v("training_set size", train_set.len()))
    Logger().write(Logger.format_k_v("validation_set size", valid_set.len()))
    #Logger().write(Logger.format_k_v("test_set size", test_set.len()
    #                                 if test_set is not None else 0))

    Logger().write("\nCalculating shift and scale from training set...\n")
    #shift, scale = train_set.shift_scale_dataset()
    shift = train_set.get_per_atom_energy_mean()
    scale = train_set.get_force_rmse()
    config.update({KEY.SHIFT: shift, KEY.SCALE: scale})
    #valid_set.shift_scale_dataset(shift=shift, scale=scale)
    Logger().write(f"calculated per_atom_energy mean shift is {shift:.6f} eV\n")
    Logger().write(f"calculated force rms scale is {scale:.6f} eV/Angstrom\n")

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
    """
    train_loader = DataLoader(train_set.to_list(), batch_size,
                              shuffle=True, num_workers=num_workers,)
    valid_loader = DataLoader(valid_set.to_list(), batch_size,
                              num_workers=num_workers,)
    """
    #if test_set is not None:
    #    test_loader = DataLoader(test_set.to_list(), batch_size)

    statistic_values = (avg_num_neigh, shift, scale)
    data_lists = (train_set.to_list(), valid_set.to_list(), test_set.to_list())

    # TODO: After I indtroduce valid_set manually, there is chance that
    #       the user_labels is not the same as the dataset.user_labels
    #       the case is not debugged!
    user_labels = dataset.user_labels.copy()
    return statistic_values, data_lists, user_labels
