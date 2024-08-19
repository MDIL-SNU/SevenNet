import csv
import os
from typing import List

import numpy as np
import torch
from ase import io
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm

import sevenn._keys as KEY
import sevenn.error_recorder as error_recorder
from sevenn.train.dataload import graph_build
from sevenn.train.dataset import AtomGraphDataset
from sevenn.util import (
    model_from_checkpoint,
    pretrained_name_to_path,
    to_atom_graph_list,
)

# TODO: use updated dataset construction scheme, not directly call graph_build


def load_sevenn_data(sevenn_datas: str, cutoff, type_map):
    full_dataset = None
    for sevenn_data in sevenn_datas:
        with open(sevenn_data, 'rb') as f:
            dataset = torch.load(f)
        if full_dataset is None:
            full_dataset = dataset
        else:
            full_dataset.augment(dataset)
    if full_dataset.cutoff != cutoff:
        raise ValueError(f'cutoff mismatch: {full_dataset.cutoff} != {cutoff}')
    if full_dataset.x_is_one_hot_idx and full_dataset.type_map != type_map:
        raise ValueError(
            "loaded dataset's x is not atomic numbers.                 this is"
            ' deprecated. Create dataset from structure list                '
            ' with the newest version of sevenn'
        )
    return full_dataset


# TODO: Outcar can be trajectory
def outcars_to_atoms(outcars: List[str]):
    atoms_list = []
    info_dct = {'data_from': 'infer_OUTCAR'}
    for outcar_path in outcars:
        atoms = io.read(outcar_path)
        info_dct_f = {**info_dct, 'file': os.path.abspath(outcar_path)}
        atoms.info = info_dct_f
        atoms_list.append(atoms)
    return atoms_list


def poscars_to_atoms(poscars: List[str]):
    """
    load poscars to ase atoms list
    dummy y values are injected for convenience
    """
    atoms_list = []
    stress_dummy = np.array([0, 0, 0, 0, 0, 0])
    calc_results = {'energy': 0, 'free_energy': 0, 'stress': stress_dummy}
    info_dct = {'data_from': 'infer_POSCAR'}
    for poscar_path in poscars:
        atoms = io.read(poscar_path)
        natoms = len(atoms.get_atomic_numbers())
        dummy_force = np.zeros((natoms, 3))
        dummy_calc_res = calc_results.copy()
        dummy_calc_res['forces'] = dummy_force
        calculator = SinglePointCalculator(atoms, **dummy_calc_res)
        atoms = calculator.get_atoms()
        info_dct_f = {**info_dct, 'file': os.path.abspath(poscar_path)}
        atoms.info = info_dct_f
        atoms_list.append(atoms)
    return atoms_list


def get_error_recorder():
    config = [
        ('Energy', 'RMSE'),
        ('Force', 'RMSE'),
        ('Stress', 'RMSE'),
        ('Energy', 'MAE'),
        ('Force', 'MAE'),
        ('Stress', 'MAE'),
    ]
    err_metrics = []
    for err_type, metric_name in config:
        metric_kwargs = error_recorder.ERROR_TYPES[err_type].copy()
        metric_kwargs['name'] += f'_{metric_name}'
        metric_cls = error_recorder.ErrorRecorder.METRIC_DICT[metric_name]
        err_metrics.append(metric_cls(**metric_kwargs))
    return error_recorder.ErrorRecorder(err_metrics)


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


def inference_main(
    checkpoint,
    fnames,
    output_path,
    num_cores=1,
    num_workers=1,
    device='cpu',
    batch_size=5,
    on_the_fly_graph_build=True,
):
    if os.path.isfile(checkpoint):
        pass
    else:
        checkpoint = pretrained_name_to_path(checkpoint)
    model, config = model_from_checkpoint(checkpoint)
    model.to(device)
    model.set_is_batch_data(True)
    model.eval()

    cutoff = config[KEY.CUTOFF]
    type_map = config[KEY.TYPE_MAP]

    head = os.path.basename(fnames[0])
    atoms_list = None
    inference_set = None
    no_ref = False
    if head.endswith('sevenn_data'):
        inference_set = load_sevenn_data(fnames, cutoff, type_map)
        on_the_fly_graph_build = False
    else:
        if head.startswith('POSCAR'):
            atoms_list = poscars_to_atoms(fnames)
            no_ref = True  # poscar has no y value
        elif head.startswith('OUTCAR'):
            atoms_list = outcars_to_atoms(fnames)
        else:
            atoms_list = []
            for fname in fnames:
                atoms_list.extend(io.read(fname, index=':'))

    if not on_the_fly_graph_build:  # old code
        from torch_geometric.loader import DataLoader
        if atoms_list is not None:
            data_list = graph_build(atoms_list, cutoff, num_cores=num_cores)
            inference_set = AtomGraphDataset(data_list, cutoff)
        assert inference_set is not None

        inference_set.x_to_one_hot_idx(type_map)
        inference_set.toggle_requires_grad_of_data(KEY.EDGE_VEC, True)
        infer_list = inference_set.to_list()
        loader = DataLoader(
            infer_list,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
        output_list = []
    else:  # new
        from torch.utils.data.dataloader import DataLoader

        from sevenn.train.collate import AtomsToGraphCollater
        collate = AtomsToGraphCollater(
            atoms_list,
            cutoff,
            type_map,
            transfer_info=True
        )
        loader = DataLoader(
            atoms_list,
            collate_fn=collate,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    recorder = get_error_recorder()
    output_list = []
    try:
        for batch in tqdm(loader):
            batch = batch.to(device, non_blocking=True)
            batch[KEY.EDGE_VEC].requires_grad_(True)
            output = model(batch)
            output.detach().to('cpu')
            recorder.update(output)
            output_list.extend(to_atom_graph_list(output))  # unroll batch data
    except Exception as e:
        print(e)
        print("Keeping 'info' failed. Try with separated info")
        recorder.epoch_forward()
        infer_list, _ = inference_set.separate_info()
        loader = DataLoader(infer_list, batch_size=batch_size, shuffle=False)
        output_list = []
        for batch in tqdm(loader):
            batch = batch.to(device, non_blocking=True)
            batch[KEY.EDGE_VEC].requires_grad_(True)
            output = model(batch)
            output.detach().to('cpu')
            recorder.update(output)
            output_list.extend(to_atom_graph_list(output))  # unroll batch data

    errors = recorder.epoch_forward()

    if not no_ref:
        with open(f'{output_path}/errors.txt', 'w') as f:
            for key, val in errors.items():
                f.write(f'{key}: {val}\n')

    write_inference_csv(output_list, output_path)
