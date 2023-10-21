from typing import Dict
import os
import sys
import random

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from sevenn.train.trainer import Trainer
from sevenn.model_build import build_E3_equivariant_model
from sevenn.scripts.processing_dataset import processing_dataset
from sevenn.scripts.processing_continue import processing_continue
from sevenn.scripts.processing_epoch import processing_epoch
from sevenn.sevenn_logger import Logger
import sevenn._keys as KEY


# TODO: E3_equivariant model assumed
# TODO: More clear logging
def train(config: Dict, working_dir: str):
    """
    Main program flow
    """
    Logger().timer_start("total")
    rank = config[KEY.RANK]
    is_ddp = config[KEY.IS_DDP]
    seed = config[KEY.RANDOM_SEED]
    batch_size = config[KEY.BATCH_SIZE]
    random.seed(seed)
    torch.manual_seed(seed)

    # config updated
    data_lists, user_labels = processing_dataset(config, working_dir)
    train, valid, test = data_lists
    if is_ddp:
        dist.barrier()
        train_sampler = DistributedSampler(train,
                                           num_replicas=dist.get_world_size(),
                                           rank=dist.get_rank(),
                                           shuffle=config[KEY.TRAIN_SHUFFLE])
        valid_sampler = DistributedSampler(valid,
                                           num_replicas=dist.get_world_size(),
                                           rank=dist.get_rank())
        train_loader = DataLoader(train, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(valid, batch_size=batch_size, sampler=valid_sampler)
    else:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=config[KEY.TRAIN_SHUFFLE])
        valid_loader = DataLoader(valid, batch_size=batch_size)
    loaders = (train_loader, valid_loader, None)

    Logger().write("\nModel building...\n")
    model = build_E3_equivariant_model(config)
    Logger().write("Model building was successful\n")

    # config updated
    if config[KEY.CONTINUE][KEY.CHECKPOINT] is not False:
        trainer = processing_continue(model, user_labels, config)
    else:
        trainer = Trainer(model, user_labels, config)

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
