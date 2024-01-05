import csv
import os
from typing import List

import numpy as np
import torch
from ase import Atoms, io
from ase.calculators.singlepoint import SinglePointCalculator
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import sevenn._keys as KEY
from sevenn._const import LossType
from sevenn.nn.node_embedding import get_type_mapper_from_specie
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.train.dataload import graph_build
from sevenn.train.dataset import AtomGraphDataset
from sevenn.util import (
    AverageNumber,
    load_model_from_checkpoint,
    postprocess_output,
    squared_error,
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
            " deprecated. Create dataset from structure list                "
            " with the newest version of sevenn"
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


def write_inference_csv(output_list, rmse_dct, out, no_ref):
    is_stress = 'STRESS' in rmse_dct
    for i, output in enumerate(output_list):
        output = output.fit_dimension()
        if is_stress:
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

    if not no_ref:
        with open(f'{out}/rmse.txt', 'w') as f:
            f.write(f"Energy rmse (eV/atom): {rmse_dct['ENERGY']}\n")
            f.write(f"Force rmse (eV/A): {rmse_dct['FORCE']}\n")
            if is_stress:
                f.write(f"Stress rmse (kbar): {rmse_dct['STRESS']}\n")

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
    checkpoint, fnames, output_path, num_cores=1, device='cpu', batch_size=5
):
    checkpoint = torch.load(checkpoint, map_location=device)
    config = checkpoint['config']
    cutoff = config[KEY.CUTOFF]
    type_map = config[KEY.TYPE_MAP]

    model = load_model_from_checkpoint(checkpoint)
    model.to(device)
    model.set_is_batch_data(True)
    model.eval()

    head = os.path.basename(fnames[0])
    atoms_list = None
    no_ref = False
    if head.endswith('sevenn_data'):
        inference_set = load_sevenn_data(fnames, cutoff, type_map)
    else:
        if head.startswith('POSCAR'):
            atoms_list = poscars_to_atoms(fnames)
            no_ref = True  # poscar has no y value
        elif head.startswith('OUTCAR'):
            atoms_list = outcars_to_atoms(fnames)
        data_list = graph_build(atoms_list, cutoff, num_cores=num_cores)
        inference_set = AtomGraphDataset(data_list, cutoff)

    inference_set.x_to_one_hot_idx(type_map)
    if config[KEY.IS_TRAIN_STRESS]:
        inference_set.toggle_requires_grad_of_data(KEY.POS, True)
        is_stress = True
    else:
        inference_set.toggle_requires_grad_of_data(KEY.EDGE_VEC, True)
        is_stress = False

    loss_types = [LossType.ENERGY, LossType.FORCE]
    if is_stress:
        loss_types.append(LossType.STRESS)

    l2_err = {k: AverageNumber() for k in loss_types}
    infer_list = inference_set.to_list()
    # infer_list, info_list = inference_set.seperate_info()
    loader = DataLoader(infer_list, batch_size=batch_size, shuffle=False)

    output_list = []
    for batch in tqdm(loader):
        batch = batch.to(device, non_blocking=True)
        output = model(batch)
        output.detach().to('cpu')
        results = postprocess_output(output, loss_types)
        for loss_type in loss_types:
            l2_err[loss_type].update(squared_error(*results[loss_type]))
        output_list.extend(to_atom_graph_list(output))  # unroll batch data

    # to more readable format
    rmse_dct = {k.name: np.sqrt(v.get()) for k, v in l2_err.items()}
    write_inference_csv(output_list, rmse_dct, output_path, no_ref=no_ref)
