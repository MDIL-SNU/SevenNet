from typing import Dict
import sys
import math
import os
import random
import copy

import torch
import torch.autograd
from torch_geometric.loader import DataLoader
import e3nn.util.jit

import sevenn
import sevenn.train
from sevenn.train.dataload import parse_structure_list
from sevenn.train.dataset import AtomGraphDataset
from sevenn.train.trainer import Trainer, DataSetType
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.scripts.deploy import deploy, deploy_from_compiled
from sevenn.scripts.plot import draw_learning_curve
from sevenn.sevenn_logger import Logger
import sevenn._keys as KEY


def init_dataset_from_structure_list(data_config, working_dir):
    cutoff = data_config[KEY.CUTOFF]
    # chemical_species = data_config[KEY.CHEMICAL_SPECIES]
    format_outputs = data_config[KEY.FORMAT_OUTPUTS]
    structure_list_files = data_config[KEY.STRUCTURE_LIST]
    model_type = data_config[KEY.MODEL_TYPE]

    type_map = data_config[KEY.TYPE_MAP]

    if model_type == 'E3_equivariant_model':
        def preprocessor(x):
            return AtomGraphData.data_for_E3_equivariant_model(x, cutoff, type_map)
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


def init_dataset(data_config, working_dir):
    full_dataset = None

    if data_config[KEY.STRUCTURE_LIST] is not False:
        Logger().write("Loading dataset from structure lists\n")
        full_dataset = init_dataset_from_structure_list(data_config, working_dir)

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


# TODO: E3_equivariant model assumed
# TODO: implement continue
def train(config: Dict, working_dir: str):
    """
    main program flow
    """
    prefix = f"{os.path.abspath(working_dir)}/"
    Logger().timer_start("total")

    seed = config[KEY.RANDOM_SEED]
    random.seed(seed)
    torch.manual_seed(seed)
    device = config[KEY.DEVICE]

    # load data set
    Logger().write("\nInitializing dataset...\n")
    try:
        dataset = init_dataset(config, working_dir)
        Logger().write("Dataset initialization was successful\n")
        natoms = dataset.get_natoms(config[KEY.TYPE_MAP])

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
        shift, scale = train_set.shift_scale_dataset()
        config.update({KEY.SHIFT: shift, KEY.SCALE: scale})
        valid_set.shift_scale_dataset(shift=shift, scale=scale)
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
            test_set.shift_scale_dataset(shift=shift, scale=scale)
            test_loader = DataLoader(test_set.to_list(), batch_size)
    except Exception as e:
        Logger().error(e)
        sys.exit(1)

    if config[KEY.CONTINUE] is None:
        Logger().write("\nModel building...\n")
        # initialize model
        try:
            model = build_E3_equivariant_model(config)
            # model.set_is_batch_data(True)
            # compile mode
            # model = e3nn.util.jit.script(model)  # compile model for speed up
        except Exception as e:
            Logger().error(e)
            sys.exit(1)
        Logger().write("Model building was successful\n")
        user_labels = dataset.user_labels
        trainer = Trainer(model, user_labels, config)
    else:
        # TODO: checkpoint validation? compatiblity with current config? shift scale?
        Logger().write("\nContinue found, loading checkpoint\n")
        checkpoint = torch.load(config[KEY.CONTINUE])
        user_labels = dataset.user_labels
        trainer = Trainer.from_checkpoint_dict(checkpoint, user_labels)
        model = trainer.model
        Logger().write("\ncheckpoint previous epoch: {checkpoint['epoch']}\n")
        Logger().write("\ncheckpoint loading was successful\n")

    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger().write(f"Total number of weight in model is {num_weights}\n")

    Logger().write("Trainer initialized. program is ready to training\n")
    Logger().bar()

    min_loss = 100

    total_epoch = config[KEY.EPOCH]
    skip_output_until = config[KEY.SKIP_OUTPUT_UNTIL]

    output_per_epoch = config[KEY.OUTPUT_PER_EPOCH]
    per_epoch = output_per_epoch[KEY.PER_EPOCH]
    per_epoch = total_epoch if per_epoch is False else per_epoch
    draw_parity = output_per_epoch[KEY.DRAW_PARITY]
    is_save_data_pickle = output_per_epoch[KEY.SAVE_DATA_PICKLE]
    is_model_check_point = output_per_epoch[KEY.MODEL_CHECK_POINT]
    is_deploy_model = output_per_epoch[KEY.DEPLOY_MODEL]

    draw_lc = config[KEY.DRAW_LC]

    def output(is_best):
        """
        by default, make all outputs if best (it will overwrite previous one)
        """
        suffix = "_best" if is_best else f"_{epoch}"

        if draw_lc:
            draw_learning_curve(loss_hist_print, f"{prefix}/learning_curve.png")
        if is_deploy_model or is_best:
            # compile mode
            # deploy_from_compiled(trainer.model,
            #                      config, f"{prefix}/deployed_model{suffix}.pt")
            deploy(trainer.model, config, f"{prefix}/deployed_model{suffix}.pt")
        if is_model_check_point or is_best:
            checkpoint = trainer.get_checkpoint_dict()
            checkpoint.update({'config': config, 'epoch': epoch})
            torch.save(checkpoint, f"{prefix}/checkpoint{suffix}.pth")
        if is_save_data_pickle or is_best:
            torch.save(loss_hist_print, f"{prefix}/loss_hist{suffix}.pth")
            torch.save(info_parity, f"{prefix}/parity_at{suffix}.pth")
        if draw_parity or is_best:
            #TODO: implement
            pass

    # copy loss_hist structure
    loss_hist = trainer.loss_hist
    loss_hist_print = copy.deepcopy(loss_hist)
    for epoch in range(1, total_epoch + 1):
        Logger().timer_start("epoch")
        Logger().bar()
        Logger().write(f"Epoch {epoch}/{total_epoch}\n")
        Logger().bar()

        t_pred_E, t_ref_E, t_pred_F, t_ref_F, t_graph_set, _ = \
            trainer.run_one_epoch(train_loader, DataSetType.TRAIN)

        v_pred_E, v_ref_E, v_pred_F, v_ref_F, v_graph_set, loss = \
            trainer.run_one_epoch(valid_loader, DataSetType.VALID)

        info_parity = {"t_pred_E": t_pred_E, "t_ref_E": t_ref_E,
                       "t_pred_F": t_pred_F, "t_ref_F": t_ref_F,
                       "v_pred_E": v_pred_E, "v_ref_E": v_ref_E,
                       "v_pred_F": v_pred_F, "v_ref_F": v_ref_F}
        # preprocess loss_hist, (mse -> scaled rmse)
        for data_set_key in [DataSetType.TRAIN, DataSetType.VALID]:
            for label in trainer.user_labels:
                loss_hist_print[data_set_key][label]['energy'].append(
                    math.sqrt(loss_hist[data_set_key][label]['energy'][-1]) * scale)
                loss_hist_print[data_set_key][label]['force'].append(
                    math.sqrt(loss_hist[data_set_key][label]['force'][-1]) * scale)

        Logger().epoch_write(loss_hist_print)
        Logger().timer_end("epoch", message=f"Epoch {epoch} elapsed")

        if epoch < skip_output_until:
            continue
        if loss < min_loss:
            min_loss = loss
            output(is_best=True)
            Logger().write(f"output written at epoch(best): {epoch}\n")
            # continue  # skip per epoch output if best is written
        if epoch % per_epoch == 0:
            output(is_best=False)
            Logger().write(f"output written at epoch: {epoch}\n")
    # deploy(best_model, config, f'{prefix}/deployed_model.pt')
    Logger().timer_end("total", message="Total wall time")
