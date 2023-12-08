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


def init_loader(train, valid, _, config):
    batch_size = config[KEY.BATCH_SIZE]
    is_ddp = config[KEY.IS_DDP]
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
    return train_loader, valid_loader, None


def log_model_info(model, config):
    # print model irreps for each layer
    Logger().write("Irreps of features\n")
    Logger().format_k_v("edge_feature",
                        model.get_irreps_in("EdgeEmbedding", "irreps_out"),
                        write=True)
    for i in range(config[KEY.NUM_CONVOLUTION]):
        Logger().format_k_v(f"{i}th node_feature",
                            model.get_irreps_in(f"{i}_self_interaction_1"),
                            write=True)
    Logger().format_k_v(f"readout irreps",
                        model.get_irreps_in(f"{i} equivariant gate", "irreps_out"),
                        write=True)
    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger().write(f"Total number of weight in model is {num_weights}\n")


# TODO: E3_equivariant model assumed
# TODO: More clear logging
def train(config, working_dir: str):
    """
    Main program flow
    """
    Logger().timer_start("total")
    seed = config[KEY.RANDOM_SEED]
    random.seed(seed)
    torch.manual_seed(seed)

    # config updated
    if config[KEY.CONTINUE][KEY.CHECKPOINT] is not False:
        state_dicts, start_epoch, init_csv = processing_continue(config)
    else:
        state_dicts, start_epoch, init_csv = None, 1, True

    # config updated
    data_lists = processing_dataset(config, working_dir)
    loaders = init_loader(*data_lists, config)

    Logger().write("\nModel building...\n")
    model = build_E3_equivariant_model(config)

    Logger().write("Model building was successful\n")

    trainer = Trainer(model, config)
    if state_dicts is not None:
        trainer.load_state_dicts(*state_dicts, strict=False)

    log_model_info(model, config)

    Logger().write("Trainer initialized, ready to training\n")
    Logger().bar()

    processing_epoch(trainer, config, loaders, start_epoch, init_csv, working_dir)
    Logger().timer_end("total", message="Total wall time")

