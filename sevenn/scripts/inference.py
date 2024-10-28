import csv
import os
import tempfile
from typing import IO, Iterable, List, Union

import ase.io
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import sevenn._keys as KEY
import sevenn.train.dataload as dl
import sevenn.util as util
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.graph_dataset import SevenNetGraphDataset


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
            cell_dct = {KEY.CELL: output[KEY.CELL]}
            cell_dct = unfold_dct_val(cell_dct, [KEY.CELL], ['a', 'b', 'c'])
            data = {
                **unfold_dct_val(output, per_graph_keys, sfx_list),
                **cell_dct,
            }
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


def _extract_unlabeled_data(targets: List[str], tmp_file: IO, **data_kwargs):
    # for only, ase readable, it may be unlabeled
    # extract such files and returns graph list built with these

    def assign_dummy_y(atoms):
        dummy = {'energy': np.nan, 'free_energy': np.nan}
        dummy['forces'] = np.full((len(atoms), 3), np.nan)  # type: ignore
        dummy['stress'] = np.full((6,), np.nan)  # type: ignore
        calc = SinglePointCalculator(atoms, **dummy)
        atoms = calc.get_atoms()
        return calc.get_atoms()

    new_targets = []
    atoms_list_patched = []
    unlabeled_file_list = []
    for target in targets:
        if not (
            not target.endswith('.pt')
            and not target.endswith('.sevenn_data')
            and 'structure_list' not in target
        ):
            new_targets.append(target)
            continue
        # it must be ase readable
        try:
            _ = dl.ase_reader(target, **data_kwargs)
            new_targets.append(target)  # No error occurred, target is labeled
        except RuntimeError or KeyError:
            # The data is not labeled
            print(
                f'{target} seems not labeled, dummy values will be used',
                flush=True,
            )
            atoms_list = ase.io.read(target, index=':')
            for atoms in atoms_list:
                atoms_patched = assign_dummy_y(atoms)
                atoms_patched.info.update({'y_is_dummy': 'Yes'})
                atoms_list_patched.append(atoms_patched)
            unlabeled_file_list.extend([target] * len(atoms_list))

    if len(atoms_list_patched) > 0:
        ase.io.write(tmp_file, atoms_list_patched, format='extxyz')
        tmp_file.flush()
        new_targets.append(tmp_file.name)
    return new_targets, unlabeled_file_list


def _patch_data_info(
    graph_list: Iterable[AtomGraphData], full_file_list: List[str]
) -> None:
    keys = set()
    for graph, path in zip(graph_list, full_file_list):
        graph[KEY.INFO].update({'file': os.path.abspath(path)})
        keys.update(graph[KEY.INFO].keys())

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
    **data_kwargs,
) -> None:
    """
    Inference model on the target dataset, writes
    per_graph, per_atom inference results in csv format
    to the output_dir
    If given target doesn't have EFS key, it puts dummy
    values to the it.

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
    model, _ = util.model_from_checkpoint(checkpoint)
    cutoff = model.cutoff

    if isinstance(targets, str):
        targets = [targets]

    full_file_list = []
    with tempfile.NamedTemporaryFile('w+') as tmp_file:
        targets, unlabeled_file_list = _extract_unlabeled_data(
            targets, tmp_file, **data_kwargs
        )
        if save_graph:
            dataset = SevenNetGraphDataset(
                cutoff=cutoff,
                root=output_dir,
                files=targets,
                process_num_cores=num_workers,
                processed_name='saved_graph.pt',
                **data_kwargs,
            )
            full_file_list = dataset.full_file_list
        else:
            dataset = []
            for file in targets:
                tmplist = SevenNetGraphDataset.file_to_graph_list(
                    filename=file,
                    cutoff=cutoff,
                    num_cores=num_workers,
                    **data_kwargs,
                )
                dataset.extend(tmplist)
                full_file_list.extend([os.path.abspath(file)] * len(tmplist))
        if len(unlabeled_file_list) > 0:
            full_file_list = full_file_list[: -len(unlabeled_file_list)]
            full_file_list.extend(unlabeled_file_list)
    assert len(full_file_list) == len(dataset)
    _patch_data_info(dataset, full_file_list)  # type: ignore
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.set_is_batch_data(True)
    model.eval()

    rec = util.get_error_recorder()
    output_list = []

    for batch in tqdm(loader):
        batch = batch.to(device)
        output = model(batch)
        rec.update(output)
        output_list.extend(util.to_atom_graph_list(output))

    errors = rec.epoch_forward()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'errors.txt'), 'w', encoding='utf-8') as f:
        for key, val in errors.items():
            f.write(f'{key}: {val}\n')

    write_inference_csv(output_list, output_dir)
