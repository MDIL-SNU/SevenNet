from typing import Any, Dict, List, Optional

import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import sevenn._keys as KEY
from sevenn.logger import Logger
from sevenn.model_build import build_E3_equivariant_model
from sevenn.scripts.processing_continue import (
    convert_modality_of_checkpoint_state_dct,
)
from sevenn.train.trainer import Trainer


def loader_from_config(
    config: Dict[str, Any], dataset: Dataset, is_train: bool = False
) -> DataLoader:
    batch_size = config[KEY.BATCH_SIZE]
    shuffle = is_train and config[KEY.TRAIN_SHUFFLE]
    sampler = None
    loader_args = {'dataset': dataset, 'batch_size': batch_size, 'shuffle': shuffle}
    if KEY.NUM_WORKERS in config and config[KEY.NUM_WORKERS] > 0:
        loader_args.update({'num_workers': config[KEY.NUM_WORKERS]})

    if config[KEY.IS_DDP]:
        dist.barrier()
        sampler = DistributedSampler(
            dataset, dist.get_world_size(), dist.get_rank(), shuffle=shuffle
        )
        loader_args.update({'sampler': sampler})
        loader_args.pop('shuffle')  # sampler is mutually exclusive with shuffle
    return DataLoader(**loader_args)


def _build_memory_loader(config: Dict[str, Any]) -> DataLoader:
    """Build the reEWC rehearsal (replay) memory loader from load_memory_path."""
    import random

    from sevenn.train.graph_dataset import SevenNetGraphDataset

    memory_paths = config.get(KEY.LOAD_MEMORY_PATH, False)
    if not memory_paths:
        raise ValueError('rehearsal is True but load_memory_path is not set')
    if isinstance(memory_paths, str):
        memory_paths = [memory_paths]
    mem_batch_size = config.get(KEY.MEM_BATCH_SIZE, 0)
    if not (isinstance(mem_batch_size, int) and mem_batch_size > 0):
        raise ValueError('rehearsal requires mem_batch_size > 0')
    mem_ratio = config.get(KEY.MEM_RATIO, 1)
    if not (0 < mem_ratio <= 1):
        raise ValueError('rehearsal requires 0 < mem_ratio <= 1')

    graphs = []
    for file in memory_paths:
        graphs.extend(
            SevenNetGraphDataset.file_to_graph_list(file, cutoff=config[KEY.CUTOFF])
        )
    if mem_ratio < 1:
        random.Random(config.get(KEY.RANDOM_SEED, 1)).shuffle(graphs)
        graphs = graphs[: int(len(graphs) * mem_ratio)]
    if len(graphs) == 0:
        raise ValueError('reEWC rehearsal memory set is empty after loading')
    Logger().writeline(
        f'Rehearsal enabled: {len(graphs)} memory graphs, '
        f'mem_batch_size={mem_batch_size}'
    )
    return DataLoader(graphs, batch_size=mem_batch_size, shuffle=True)


def train_v2(config: Dict[str, Any], working_dir: str) -> None:
    """
    Main program flow, since v0.9.6
    """
    import sevenn.train.atoms_dataset as atoms_dataset
    import sevenn.train.graph_dataset as graph_dataset
    import sevenn.train.modal_dataset as modal_dataset

    from .processing_continue import processing_continue_v2
    from .processing_epoch import processing_epoch_v2

    log = Logger()
    log.timer_start('total')

    if KEY.LOAD_TRAINSET not in config and KEY.LOAD_DATASET in config:
        log.writeline('***************************************************')
        log.writeline('For train_v2, please use load_trainset_path instead')
        log.writeline('I will assign load_trainset as load_dataset')
        log.writeline('***************************************************')
        config[KEY.LOAD_TRAINSET] = config.pop(KEY.LOAD_DATASET)

    # reEWC keys are optional; with rehearsal off and no Fisher/opt set, the
    # branches below are skipped.
    rehearsal = config.get(KEY.REHEARSAL, False)
    _cont = config.get(KEY.CONTINUE, {})
    ewc_active = bool(_cont.get(KEY.FISHER, False)) or bool(
        _cont.get(KEY.OPT_PARAMS, False)
    )
    memory_paths = config.get(KEY.LOAD_MEMORY_PATH, False)
    if (rehearsal or ewc_active) and config.get(KEY.IS_DDP, False):
        raise NotImplementedError(
            'reEWC (rehearsal/EWC) does not support distributed training'
        )
    if memory_paths and not rehearsal:
        raise ValueError(
            'load_memory_path is set but rehearsal is False; load_memory_path '
            'is reserved for reEWC rehearsal'
        )
    if rehearsal and config.get(KEY.DATASET_TYPE) == 'atoms':
        raise NotImplementedError(
            'reEWC rehearsal supports dataset_type="graph" only'
        )

    # config updated
    start_epoch = 1
    state_dicts: Optional[List[dict]] = None
    if config[KEY.CONTINUE][KEY.CHECKPOINT]:
        state_dicts, start_epoch = processing_continue_v2(config)

    if config.get(KEY.USE_MODALITY, False):
        if rehearsal or ewc_active:
            raise ValueError(
                'reEWC (rehearsal/EWC) supports single-modal models only; '
                'multifidelity/modal models are not supported'
            )
        datasets = modal_dataset.from_config(config, working_dir)
    elif config[KEY.DATASET_TYPE] == 'graph':
        # exclude the rehearsal memory set from normal dataset discovery so it
        # is not run as an extra (validation-style) loader every epoch.
        dataset_keys = None
        if memory_paths:
            dataset_keys = [
                k
                for k in config
                if k.startswith('load_')
                and k.endswith('_path')
                and k != KEY.LOAD_MEMORY_PATH
            ]
        datasets = graph_dataset.from_config(
            config, working_dir, dataset_keys=dataset_keys
        )
    elif config[KEY.DATASET_TYPE] == 'atoms':
        datasets = atoms_dataset.from_config(config, working_dir)
    else:
        raise ValueError(f'Unknown dataset type: {config[KEY.DATASET_TYPE]}')
    loaders = {
        k: loader_from_config(config, v, is_train=(k == 'trainset'))
        for k, v in datasets.items()
    }

    memory_loader = _build_memory_loader(config) if rehearsal else None

    log.write('\nModel building...\n')
    model = build_E3_equivariant_model(config)
    log.print_model_info(model, config)

    trainer = Trainer.from_config(model, config, memory_loader=memory_loader)
    if state_dicts:
        trainer.load_state_dicts(*state_dicts, strict=False)

    processing_epoch_v2(
        config, trainer, loaders, start_epoch, working_dir=working_dir
    )
    log.timer_end('total', message='Total wall time')


def train(config, working_dir: str):
    """
    Main program flow, until v0.9.5
    """
    from .processing_continue import processing_continue
    from .processing_dataset import processing_dataset
    from .processing_epoch import processing_epoch

    log = Logger()
    log.timer_start('total')

    # config updated
    state_dicts: Optional[List[dict]] = None
    if config[KEY.CONTINUE][KEY.CHECKPOINT]:
        state_dicts, start_epoch, init_csv = processing_continue(config)
    else:
        start_epoch, init_csv = 1, True

    # config updated
    train, valid, _ = processing_dataset(config, working_dir)
    datasets = {'dataset': train, 'validset': valid}
    loaders = {
        k: loader_from_config(config, v, is_train=(k == 'dataset'))
        for k, v in datasets.items()
    }
    loaders = list(loaders.values())

    log.write('\nModel building...\n')
    model = build_E3_equivariant_model(config)

    log.write('Model building was successful\n')

    trainer = Trainer.from_config(model, config)
    if state_dicts:
        state_dicts = convert_modality_of_checkpoint_state_dct(config, state_dicts)
        trainer.load_state_dicts(*state_dicts, strict=False)

    log.print_model_info(model, config)

    Logger().write('Trainer initialized, ready to training\n')
    Logger().bar()
    log.write('Trainer initialized, ready to training\n')
    log.bar()

    processing_epoch(trainer, config, loaders, start_epoch, init_csv, working_dir)
    log.timer_end('total', message='Total wall time')
