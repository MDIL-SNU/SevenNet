import random
from typing import Any, Dict, List, Optional

from torch_geometric.loader import DataLoader

import sevenn._keys as KEY
from sevenn.logger import Logger


def validate_reewc_config(config: Dict[str, Any]) -> None:
    """Fail-loud guards for reEWC. No-op when no reEWC keys are set."""
    rehearsal = config.get(KEY.REHEARSAL, False)
    cont = config.get(KEY.CONTINUE, {})
    ewc_active = bool(cont.get(KEY.FISHER, False)) or bool(
        cont.get(KEY.OPT_PARAMS, False)
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
    if (rehearsal or ewc_active) and config.get(KEY.USE_MODALITY, False):
        raise ValueError(
            'reEWC (rehearsal/EWC) supports single-modal models only; '
            'multifidelity/modal models are not supported'
        )


def reewc_dataset_keys(config: Dict[str, Any]) -> Optional[List[str]]:
    """Dataset keys for normal discovery, excluding the reserved memory set so
    it is not run as an extra (validation-style) loader every epoch. Returns
    None (load all) when no rehearsal memory set is configured."""
    memory_paths = config.get(KEY.LOAD_MEMORY_PATH, False)
    if not memory_paths:
        return None
    return [
        k
        for k in config
        if k.startswith('load_')
        and k.endswith('_path')
        and k != KEY.LOAD_MEMORY_PATH
    ]


def build_memory_loader(config: Dict[str, Any]) -> DataLoader:
    """Build the reEWC rehearsal (replay) memory loader from load_memory_path."""
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
