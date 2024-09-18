import os
from typing import Optional

from sevenn.sevenn_logger import Logger
from sevenn.train.dataload import file_to_dataset, match_reader
from sevenn.train.dataset import AtomGraphDataset


def dataset_finalize(
    dataset, labels, metadata, out, save_by_label=False, verbose=True
):
    """
    Common finalization of dataset include logging and saving
    """
    natoms = dataset.get_natoms()
    species = dataset.get_species()
    metadata = {
        **metadata,
        'labels': labels,
        'natoms': natoms,
        'species': species,
    }
    dataset.meta = metadata

    if save_by_label:
        out = os.path.dirname(out)
    elif os.path.isdir(out) and save_by_label is False:
        out = os.path.join(out, 'graph_built.sevenn_data')
    elif out.endswith('.sevenn_data') is False:
        out = out + '.sevenn_data'

    logger = Logger()
    if verbose:
        logger.writeline('The metadata of the dataset is...')
        for k, v in metadata.items():
            logger.format_k_v(k, v, write=True)
    dataset.save(out, save_by_label)
    logger.writeline(f'dataset is saved to {out}')

    return dataset


def build_script(
    source: str,
    cutoff: float,
    num_cores: int,
    label_by: str,
    out: str,
    save_by_label: bool,
    fmt: str,
    suffix: str,
    transfer_info: bool,
    metadata: Optional[dict] = None,
    **fmt_kwargs,
):
    if metadata is None:
        metadata = {}
    reader, rmeta = match_reader(fmt, **fmt_kwargs)
    metadata.update(**rmeta)
    dataset = AtomGraphDataset({}, cutoff)

    logger = Logger()
    if os.path.isdir(source):
        logger.writeline(f'Look for source dir: {source}')
        if suffix is not None:
            logger.writeline(f'Try to read files if it ends with {suffix}')
        for file in os.listdir(source):
            label = file.split('.')[0] if label_by == 'auto' else label_by
            file = os.path.join(source, file)
            if suffix is not None and file.endswith(suffix) is False:
                continue
            logger.writeline(f'Read from file: {file}')
            logger.timer_start('graph_build')
            db = file_to_dataset(
                file, cutoff, num_cores, reader, label, transfer_info
            )
            dataset.augment(db)
            logger.timer_end('graph_build', f'{label} graph build time')
    elif os.path.isfile(source):
        file = source
        label = file.split('.')[0] if label_by == 'auto' else label_by
        logger.writeline(f'Read from file: {file}')
        logger.timer_start('graph_build')
        db = file_to_dataset(
            file, cutoff, num_cores, reader, label, transfer_info
        )
        dataset.augment(db)
        logger.timer_end('graph_build', f'{label} graph build time')
    else:
        raise ValueError(f'source {source} is not a file or dir')

    dataset_finalize(dataset, label, metadata, out, save_by_label)
