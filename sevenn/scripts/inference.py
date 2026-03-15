import csv
import os
from typing import Iterable, List, Optional, Union

import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn.train.modal_dataset import SevenNetMultiModalDataset


def write_inference_csv(output_list, out):
    for i, output in enumerate(output_list):
        output = output.fit_dimension()
        output[KEY.STRESS] = output[KEY.STRESS] * 1602.1766208
        output[KEY.PRED_STRESS] = output[KEY.PRED_STRESS] * 1602.1766208
        output_list[i] = output.to_numpy_dict()

    per_graph_keys = [
        KEY.NUM_ATOMS,
        KEY.USER_LABEL,
        KEY.ENERGY,
        KEY.PRED_TOTAL_ENERGY,
        KEY.STRESS,
        KEY.PRED_STRESS,
    ]

    per_atom_keys = [
        KEY.ATOMIC_NUMBERS,
        KEY.ATOMIC_ENERGY,
        KEY.POS,
        KEY.FORCE,
        KEY.PRED_FORCE,
    ]

    def unfold_dct_val(dct, keys, suffix_list=None):
        res = {}
        if suffix_list is None:
            suffix_list = range(100)
        for k in keys:
            if k not in dct:
                res[k] = '-'
            elif isinstance(dct[k], np.ndarray) and dct[k].ndim != 0:
                res.update(
                    {f'{k}_{suffix_list[i]}': v for i, v in enumerate(dct[k])}
                )
            else:
                res[k] = dct[k]
        return res

    def per_atom_dct_list(dct, keys):
        sfx_list = ['x', 'y', 'z']
        res = []
        natoms = dct[KEY.NUM_ATOMS]
        extracted = {k: dct[k] for k in keys}
        for i in range(natoms):
            raw = {}
            raw.update({k: v[i] for k, v in extracted.items()})
            per_atom_dct = unfold_dct_val(raw, keys, suffix_list=sfx_list)
            res.append(per_atom_dct)
        return res

    try:
        with open(f'{out}/info.csv', 'w', newline='') as f:
            header = output_list[0][KEY.INFO].keys()
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for output in output_list:
                writer.writerow(output[KEY.INFO])
    except (KeyError, TypeError, AttributeError, csv.Error) as e:
        print(e)
        print('failed to write meta data, info.csv is not written')

    with open(f'{out}/per_graph.csv', 'w', newline='') as f:
        sfx_list = ['xx', 'yy', 'zz', 'xy', 'yz', 'zx']  # for stress
        writer = None
        for output in output_list:
            data = unfold_dct_val(output, per_graph_keys, sfx_list)

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                writer.writeheader()
            writer.writerow(data)

    with open(f'{out}/per_atom.csv', 'w', newline='') as f:
        writer = None
        for i, output in enumerate(output_list):
            list_of_dct = per_atom_dct_list(output, per_atom_keys)
            for j, dct in enumerate(list_of_dct):
                idx_dct = {'stct_id': i, 'atom_id': j}
                data = {**idx_dct, **dct}
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    writer.writeheader()
                writer.writerow(data)


def _patch_data_info(
    graph_list: Iterable[AtomGraphData], full_file_list: List[str]
) -> None:
    keys = set()
    for graph, path in zip(graph_list, full_file_list):
        if KEY.INFO not in graph:
            graph[KEY.INFO] = {}
        graph[KEY.INFO].update({'file': os.path.abspath(path)})
        keys.update(graph[KEY.INFO].keys())

    # save only safe subset of info (for batching)
    for graph in graph_list:
        info_dict = graph[KEY.INFO]
        info_dict.update({k: '' for k in keys if k not in info_dict})


def inference(
    checkpoint: str,
    targets: Union[str, List[str]],
    output_dir: str,
    num_workers: int = 1,
    device: str = 'cpu',
    batch_size: int = 4,
    save_graph: bool = False,
    allow_unlabeled: bool = False,
    modal: Optional[str] = None,
    enable_cueq: bool = False,
    enable_flash: bool = False,
    enable_oeq: bool = False,
    **data_kwargs,
) -> None:
    """
    Inference model on the target dataset, writes
    per_graph, per_atom inference results in csv format
    to the output_dir
    If a given target doesn't have EFS key, it puts dummy
    values.

    Args:
        checkpoint: model checkpoint path,
        target: path, or list of path to evaluate. Supports
            ASE readable, sevenn_data/*.pt, .sevenn_data, and
            structure_list
        output_dir: directory to write results
        num_workers: number of workers to build graph
        device: device to evaluate, defaults to 'auto'
        batch_size: batch size for inference
        save_grpah: if True, save preprocessed graph to output dir
        data_kwargs: keyword arguments used when reading targets,
            for example, given index='-1', only the last snapshot
            will be evaluated if it was ASE readable.
            While this function can handle different types of targets
            at once, it will not work smoothly with data_kwargs

    """
    # TODO: False as default, priority?
    model, _ = util.model_from_checkpoint(
        checkpoint,
        enable_cueq=enable_cueq or None,
        enable_flash=enable_flash or None,
        enable_oeq=enable_oeq or None,
    )
    cutoff = model.cutoff

    if modal:
        if model.modal_map is None:
            raise ValueError('Modality given, but model has no modal_map')
        if modal not in model.modal_map:
            _modals = list(model.modal_map.keys())
            raise ValueError(f'Unknown modal {modal} (not in {_modals})')

    if isinstance(targets, str):
        targets = [targets]

    full_file_list = []
    if save_graph:
        dataset = SevenNetGraphDataset(
            cutoff=cutoff,
            root=output_dir,
            files=targets,
            process_num_cores=num_workers,
            processed_name='saved_graph.pt',
            **data_kwargs,
        )
        full_file_list = dataset.full_file_list  # TODO: not used currently
    else:
        dataset = []
        for file in targets:
            tmplist = SevenNetGraphDataset.file_to_graph_list(
                file,
                cutoff=cutoff,
                num_cores=num_workers,
                allow_unlabeled=allow_unlabeled,
                **data_kwargs,
            )
            dataset.extend(tmplist)
            full_file_list.extend([os.path.abspath(file)] * len(tmplist))
    if (
        full_file_list is not None
        and len(full_file_list) == len(dataset)
        and not isinstance(dataset, SevenNetGraphDataset)
    ):
        _patch_data_info(dataset, full_file_list)  # type: ignore

    if modal:
        dataset = SevenNetMultiModalDataset({modal: dataset})  # type: ignore

    loader = DataLoader(dataset, batch_size, shuffle=False)  # type: ignore

    model.to(device)
    model.set_is_batch_data(True)
    model.eval()

    rec = util.get_error_recorder()
    output_list = []

    for batch in tqdm(loader):
        batch = batch.to(device)
        output = model(batch).detach().cpu()
        rec.update(output)
        output_list.extend(util.to_atom_graph_list(output))

    errors = rec.epoch_forward()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'errors.txt'), 'w', encoding='utf-8') as f:
        for key, val in errors.items():
            f.write(f'{key}: {val}\n')

    write_inference_csv(output_list, output_dir)
