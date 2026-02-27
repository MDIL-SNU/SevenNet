import math
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
    is_train = dataset_key == 'trainset'
    batch_size = config[KEY.BATCH_SIZE]

    if isinstance(dataset, dict):
        batch_size = dataset.get('batch_size', batch_size)
        dataset = dataset['dataset']

    shuffle = is_train and config[KEY.TRAIN_SHUFFLE]
    train_by_batch = config.get(KEY.TRAIN_BY_BATCH, False)
    sampler = None

    loader_args = {'dataset': dataset, 'batch_size': batch_size, 'shuffle': shuffle}
    if KEY.NUM_WORKERS in config and config[KEY.NUM_WORKERS] > 0:
        loader_args.update({'num_workers': config[KEY.NUM_WORKERS]})

    if (loader_kwargs := config.get(KEY.LOADER_KWARGS, None)) is not None:
        loader_args.update(**loader_kwargs)

    if config[KEY.IS_DDP]:
        dist.barrier()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
        loader_args.update({'sampler': sampler})
        loader_args.pop('shuffle')  # sampler is mutually exclusive with shuffle
    else:
        world_size, rank = 1, 0

    # Use OrderedSampler for batch training mode to preserve data order
    if train_by_batch and is_train:
        from sevenn.train.sampler import OrderedSampler

        seed = config.get(KEY.RANDOM_SEED, None)
        try:
            sequence = config[f'load_{dataset_key}_sequence']['total_sequence_path']
        except:
            sequence = None
        if sequence is not None:  # when using custom sequence (e.g. subset)
            sequence = np.load(sequence)
        sampler = OrderedSampler(dataset, sequence, shuffle, seed, world_size, rank)
        loader_args.update({'sampler': sampler})
        loader_args.pop(
            'shuffle', None
        )  # sampler is mutually exclusive with shuffle

    return DataLoader(**loader_args)


def update_config_for_batch_training(
    config: Dict[str, Any],
    train_loader
) -> None:
    """
    Update scheduler parameters for batch-level training.

    This converts epoch-based scheduler parameters to step-based parameters
    when using batch training mode.
    """
    if not config.get(KEY.TRAIN_BY_BATCH, False):
        return

    # convert float type `epoch` related parameters for batch training
    effective_batch_size = config[KEY.WORLD_SIZE] * config[KEY.BATCH_SIZE]
    steps_per_epoch = math.ceil(
        train_loader.sampler.total_size / effective_batch_size
    )

    scheduler_type = config.get(KEY.SCHEDULER, 'exponentiallr').lower()
    scheduler_param = config.get(KEY.SCHEDULER_PARAM, {})
    config[KEY.SCHEDULER_BATCH_MODE] = scheduler_param.pop(
        KEY.SCHEDULER_BATCH_MODE, False
    )

    if scheduler_type == 'onecyclelr':  # special case, always batch mode
        total_steps = scheduler_param.get('total_steps', None)
        if total_steps is None:
            # total_steps not given, automatically calculated
            # allow epochs to be float for SWA
            epochs = scheduler_param.get('epochs', None)
            if epochs is None:
                raise ValueError('One of total_steps or epochs should be given')
            total_steps = math.ceil(epochs * steps_per_epoch)
        config[KEY.SCHEDULER_PARAM]['total_steps'] = total_steps
        config[KEY.SCHEDULER_BATCH_MODE] = True

    elif config[KEY.SCHEDULER_BATCH_MODE]:
        scheduler_epoch_params = {
            'linearlr': ['total_iters', lambda x, y: math.ceil(x * y)],
            'cosineannealinglr': ['T_max', lambda x, y: math.ceil(x * y)],
            'exponentiallr': ['gamma', lambda x, y: x ** (1 / y)],
        }.get(scheduler_type, None)
        if scheduler_epoch_params is None:
            raise NotImplementedError(
                f'Scheduler batch mode not implemented for {scheduler_type}.'
            )

        config[KEY.SCHEDULER_PARAM][scheduler_epoch_params[0]] = (
            scheduler_epoch_params[1](
                config[KEY.SCHEDULER_PARAM][scheduler_epoch_params[0]],
                steps_per_epoch,
            )
        )


def datasets_from_py(config, script):
    import importlib.util
    from pathlib import Path

    if isinstance(script, list):
        assert len(script) == 1, 'Need single python script'
    script = script[0]

    file_path = Path(script).resolve()
    print(f'Init dataset from {file_path}', flush=True)
    spec = importlib.util.spec_from_file_location('dataset', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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

    from .processing_by_batch import processing_by_batch
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

    # Initialize data progress for batch training
    train_by_batch = config.get(KEY.TRAIN_BY_BATCH, False)
    if train_by_batch:
        data_progress = {
            KEY.TOTAL_DATA_NUM: -1,
            KEY.CURRENT_DATA_IDX: 0,
            KEY.NUMPY_RNG_STATE: None,
        }
    else:
        data_progress = {}   # dummy

    # config updated
    start_epoch = 1
    state_dicts: Optional[List[dict]] = None
    if config[KEY.CONTINUE][KEY.CHECKPOINT]:
        result = processing_continue_v2(config)
        if train_by_batch:
            state_dicts, start_epoch, data_progress = result
        else:
            state_dicts, start_epoch = result

    # Load datasets based on type
    dataset_type = config[KEY.DATASET_TYPE]
    if (
        config.get(KEY.USE_MODALITY, False)
        and not config[KEY.DATASET_TYPE] == 'custom'
    ):
        datasets = modal_dataset.from_config(config, working_dir)
    elif dataset_type == 'graph':
        datasets = graph_dataset.from_config(config, working_dir)
    elif dataset_type == 'atoms':
        datasets = atoms_dataset.from_config(config, working_dir)
    elif dataset_type == 'aselmdb':
        datasets = aselmdb_dataset.from_config(config, working_dir)
    elif dataset_type == 'custom':
        datasets = datasets_from_py(config, config.get('load_trainset_path'))
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')

    loaders = {
        k: loader_from_config(config, v, dataset_key=k)
        for k, v in datasets.items()
    }

    # Update scheduler config for batch training
    if train_by_batch:
        update_config_for_batch_training(config, loaders['trainset'])

    log.write('\nModel building...\n')
    model = build_E3_equivariant_model(config)
    log.print_model_info(model, config)

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
        k: loader_from_config(config, v, dataset_key=k)
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
