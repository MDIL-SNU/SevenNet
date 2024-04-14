import random

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import sevenn._keys as KEY
from sevenn.model_build import build_E3_equivariant_model
from sevenn.scripts.processing_continue import processing_continue
from sevenn.scripts.processing_dataset import processing_dataset
from sevenn.scripts.processing_epoch import processing_epoch
from sevenn.sevenn_logger import Logger
from sevenn.train.trainer import Trainer


def init_loaders(train, valid, _, config):
    batch_size = config[KEY.BATCH_SIZE]
    is_ddp = config[KEY.IS_DDP]
    if is_ddp:
        dist.barrier()
        train_sampler = DistributedSampler(
            train,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=config[KEY.TRAIN_SHUFFLE],
        )
        valid_sampler = DistributedSampler(
            valid, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )
        train_loader = DataLoader(
            train, batch_size=batch_size, sampler=train_sampler
        )
        valid_loader = DataLoader(
            valid, batch_size=batch_size, sampler=valid_sampler
        )
    else:
        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=config[KEY.TRAIN_SHUFFLE]
        )
        valid_loader = DataLoader(valid, batch_size=batch_size)
    return train_loader, valid_loader, None


# TODO: E3_equivariant model assumed
def train(config, working_dir: str):
    """
    Main program flow
    """
    Logger().timer_start('total')
    seed = config[KEY.RANDOM_SEED]
    random.seed(seed)
    torch.manual_seed(seed)

    # config updated
    if config[KEY.CONTINUE][KEY.CHECKPOINT] is not False:
        state_dicts, start_epoch, init_csv = processing_continue(config)
    else:
        state_dicts, start_epoch, init_csv = None, 1, True

    # config updated
    # Note that continue and dataset cannot be seperated completely
    data_lists = processing_dataset(config, working_dir)
    loaders = init_loaders(*data_lists, config)

    Logger().write('\nModel building...\n')
    model = build_E3_equivariant_model(config)

    Logger().write('Model building was successful\n')

    trainer = Trainer(model, config)
    if state_dicts is not None:
        trainer.load_state_dicts(*state_dicts, strict=False)

    Logger().print_model_info(model, config)
    # log_model_info(model, config)

    Logger().write('Trainer initialized, ready to training\n')
    Logger().bar()

    processing_epoch(
        trainer, config, loaders, start_epoch, init_csv, working_dir
    )
    Logger().timer_end('total', message='Total wall time')
