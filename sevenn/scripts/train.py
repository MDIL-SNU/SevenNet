from typing import Dict
import os
import sys
import random

import torch

from sevenn.train.trainer import Trainer
from sevenn.model_build import build_E3_equivariant_model
from sevenn.scripts.processing_dataset import processing_dataset
from sevenn.scripts.processing_continue import processing_continue
from sevenn.scripts.processing_epoch import processing_epoch
from sevenn.sevenn_logger import Logger
import sevenn._keys as KEY


# TODO: E3_equivariant model assumed
def train(config: Dict, working_dir: str):
    """
    Main program flow
    """
    Logger().timer_start("total")
    seed = config[KEY.RANDOM_SEED]
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        # config updated
        statistic_values, loaders, user_labels = \
            processing_dataset(config, working_dir)

        #avg_num_neigh, shift, scale = statistic_values
        #train_loader, valid_loader, test_loader = loaders

        Logger().write("\nModel building...\n")
        model = build_E3_equivariant_model(config)
        Logger().write("Model building was successful\n")

        optimizer_state_dict, scheduler_state_dict = None, None

        # config updated
        if config[KEY.CONTINUE][KEY.CHECKPOINT] is not False:
            optimizer_state_dict, scheduler_state_dict = \
                processing_continue(config, model, statistic_values)

        trainer = Trainer(
            model, user_labels, config,
            optimizer_state_dict=optimizer_state_dict,
            scheduler_state_dict=scheduler_state_dict
        )

        num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
        Logger().write(f"Total number of weight in model is {num_weights}\n")
        Logger().write("Trainer initialized. The program is ready to training\n")

        Logger().write("Note that...\n")
        Logger().write("Energy unit of rmse: eV/atom\n")
        Logger().write("Force unit of rmse: eV/Angstrom\n")
        Logger().write("Stress unit of rmse: kB\n")

        Logger().bar()

        processing_epoch(trainer, config, loaders, working_dir)
        Logger().timer_end("total", message="Total wall time")
    except Exception as e:
        Logger().error(e)
        sys.exit(1)


"""
def inference_poscar(config: Dict, working_dir: str):  # This is only for debugging
    prefix = f"{os.path.abspath(working_dir)}/"
    is_stress = (config[KEY.IS_TRACE_STRESS] or config[KEY.IS_TRAIN_STRESS])
    device = config[KEY.DEVICE]

    dataset = init_dataset(config, working_dir, is_stress, True)

    loader = DataLoader(dataset.to_list(), 1, shuffle=False)

    checkpoint = torch.load('checkpoint_20.pth')
    old_config = checkpoint['config']

    config.update({KEY.SHIFT: old_config[KEY.SHIFT],
                   KEY.SCALE: old_config[KEY.SCALE],
                   KEY.AVG_NUM_NEIGHBOR: old_config[KEY.AVG_NUM_NEIGHBOR]})

    for key, value in old_config.items():
        if key not in config.keys():
            print(f'{key} does not exist')
        elif config[key] != value:
            print(f'{key} is updated')

    model = build_E3_equivariant_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    for idx, data in enumerate(loader):
        data.to(device)

        result = model(data)

        if idx == 0:
            print(result[KEY.PRED_TOTAL_ENERGY])
            print(result[KEY.PRED_FORCE])
            print(result[KEY.SCALED_STRESS])
            break
"""
