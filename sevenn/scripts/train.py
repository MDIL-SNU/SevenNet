import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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
from sevenn.train.reewc.rehearsal import (
    build_memory_loader,
    reewc_dataset_keys,
    validate_reewc_config,
)
from sevenn.train.reewc.trainer import ReewcTrainer
from sevenn.train.sampler import OrderedSampler
from sevenn.train.trainer import Trainer


def loader_from_config(
    config: Dict[str, Any],
    dataset: Dataset,
    dataset_key: str,
) -> DataLoader:
    """
    Create DataLoader from config.

    Args:
        config: Configuration dictionary
        dataset: Dataset to create loader for
                 (or dict with 'dataset' and 'batch_size')
        dataset_key: Key identifying the dataset
    """
    batch_size = config[KEY.BATCH_SIZE]

    if isinstance(dataset, dict):
        batch_size = dataset.get('batch_size', batch_size)
        dataset = dataset['dataset']

    shuffle = config[KEY.TRAIN_SHUFFLE]
    sampler = None

    loader_args = {'dataset': dataset, 'batch_size': batch_size, 'shuffle': shuffle}
    if KEY.NUM_WORKERS in config and config[KEY.NUM_WORKERS] > 0:
        loader_args.update({'num_workers': config[KEY.NUM_WORKERS]})

    if (loader_kwargs := config.get(KEY.LOADER_KWARGS, None)) is not None:
        loader_args.update(**loader_kwargs)

    world_size, rank = 1, 0
    if config[KEY.IS_DDP]:
        dist.barrier()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
        loader_args.update({'sampler': sampler})
        loader_args.pop('shuffle')  # sampler is mutually exclusive with shuffle

    # Use OrderedSampler for batch training mode to preserve data order
    # verified only for validset
    # TODO: I think 'train_by_batch' and 'sampling validset' is independent,
    #       so 'train_by_batch' should be removed
    if config.get(KEY.TRAIN_BY_BATCH, False):
        sequence = config.get(f'load_{dataset_key}_sequence', {}).get(
            'total_sequence_path', None
        )

        sampler = OrderedSampler(
            dataset=dataset,
            sequence=np.load(sequence) if sequence else None,
            shuffle=shuffle,
            seed=config.get(KEY.RANDOM_SEED, 777),
            world_size=world_size,
            rank=rank,
        )
        # sampler is mutually exclusive with shuffle
        loader_args.update({'sampler': sampler, 'shuffle': None})
    return DataLoader(**loader_args)


def datasets_from_py(config, script):
    if isinstance(script, list):
        assert len(script) == 1, 'Need single python script'
    script = script[0]

    file_path = Path(script).resolve()
    print(f'Init dataset from {file_path}', flush=True)
    spec = importlib.util.spec_from_file_location('dataset', file_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore

    ret = module.dataset(config)
    assert isinstance(ret, dict) and 'trainset' in ret
    return ret


# TODO: check backward compatibility this part (batch vs. epoch)
def train_v2(config: Dict[str, Any], working_dir: str) -> None:
    """
    Main program flow, since v0.9.6

    Supports:
    - Epoch-level training (default)
    - Batch-level training (train_by_batch: true)
    """
    import sevenn.train.aselmdb_dataset as aselmdb_dataset
    import sevenn.train.atoms_dataset as atoms_dataset
    import sevenn.train.graph_dataset as graph_dataset
    import sevenn.train.modal_dataset as modal_dataset

    from .processing_by_batch import (
        processing_by_batch,
        update_config_for_batch_training,
    )
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

    validate_reewc_config(config)
    # Initialize data progress for batch training
    train_by_batch = config.get(KEY.TRAIN_BY_BATCH, False)

    data_progress = {}
    if train_by_batch:
        data_progress = {
            KEY.TOTAL_DATA_NUM: -1,
            KEY.CURRENT_DATA_IDX: 0,
            KEY.NUMPY_RNG_STATE: None,
        }

    # config updated
    start_epoch = 1
    state_dicts: Optional[List[dict]] = None
    if config[KEY.CONTINUE][KEY.CHECKPOINT]:
        # data_progress is non-empty only if train_by_batch is True
        state_dicts, start_epoch, data_progress = processing_continue_v2(config)

    # Load datasets based on type
    dataset_type = config[KEY.DATASET_TYPE]
    if (
        config.get(KEY.USE_MODALITY, False)
        and not config[KEY.DATASET_TYPE] == 'custom'
    ):
        datasets = modal_dataset.from_config(config, working_dir)
    elif dataset_type == 'graph':
        datasets = graph_dataset.from_config(
            config, working_dir, dataset_keys=reewc_dataset_keys(config)
        )
    elif dataset_type == 'atoms':
        datasets = atoms_dataset.from_config(config, working_dir)
    elif dataset_type == 'aselmdb':
        datasets = aselmdb_dataset.from_config(config, working_dir)
    elif dataset_type == 'custom':
        datasets = datasets_from_py(config, config.get('load_trainset_path'))
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')

    loaders = {
        k: loader_from_config(config, v, dataset_key=k) for k, v in datasets.items()
    }

    # Update scheduler config for batch training
    if train_by_batch:
        update_config_for_batch_training(config, loaders)

    log.write('\nModel building...\n')
    model = build_E3_equivariant_model(config)
    log.print_model_info(model, config)

    if config.get(KEY.REHEARSAL, False):
        memory_loader = build_memory_loader(config)
        trainer = ReewcTrainer.from_config(
            model, config, memory_loader=memory_loader
        )
    else:
        trainer = Trainer.from_config(model, config)

    if state_dicts:
        trainer.load_state_dicts(*state_dicts, strict=False)

    if train_by_batch:
        processing_by_batch(
            config,
            trainer,
            loaders,
            data_progress,
            start_epoch,
            working_dir=working_dir,
        )
    else:
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
        k: loader_from_config(config, v, dataset_key=k) for k, v in datasets.items()
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
