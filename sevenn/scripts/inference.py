import os
import csv
from typing import List

import torch
import numpy as np

from tqdm import tqdm

from ase import io, Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from sevenn.train.dataset import AtomGraphDataset
from sevenn.nn.node_embedding import get_type_mapper_from_specie
from sevenn.util import load_model_from_checkpoint
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.scripts.graph_build import graph_build
import sevenn._keys as KEY


def load_sevenn_datas(sevenn_datas: str, cutoff, type_map):
    full_dataset = None
    for sevenn_data in sevenn_datas:
        with open(sevenn_data, "rb") as f:
            dataset = torch.load(f)
        if full_dataset is None:
            full_dataset = dataset
        else:
            full_dataset.augment(dataset)
    if full_dataset.cutoff != cutoff:
        raise ValueError(f"cutoff mismatch: {full_dataset.cutoff} != {cutoff}")
    if full_dataset.x_is_one_hot_idx and full_dataset.type_map != type_map:
        raise ValueError("loaded dataset's x is not atomic numbers. \
                this is deprecated. Create dataset from structure list \
                with the newest version of sevenn")
    return full_dataset


#TODO: Outcar can be trajectory
def outcars_to_atoms(outcars: List[str]):
    atoms_list = []
    info_dct = {"data_from": "infer_OUTCAR"}
    for outcar_path in outcars:
        atoms = io.read(outcar_path)
        info_dct_f = {**info_dct, "file": os.path.abspath(outcar_path)}
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
    calc_results = {"energy": 0,
                    "free_energy": 0,
                    "stress": stress_dummy}
    info_dct = {"data_from": "infer_POSCAR"}
    for poscar_path in poscars:
        atoms = io.read(poscar_path)
        natoms = len(atoms.get_atomic_numbers())
        dummy_force = np.zeros((natoms, 3))
        dummy_calc_res = calc_results.copy()
        dummy_calc_res["forces"] = dummy_force
        calculator = SinglePointCalculator(atoms, **dummy_calc_res)
        atoms = calculator.get_atoms()
        info_dct_f = {**info_dct, "file": os.path.abspath(poscar_path)}
        atoms.info = info_dct_f
        atoms_list.append(atoms)
    return atoms_list


def _write_inference_csv(pred_ref_dct, rmse_dct, output_path, no_ref):
    """
    subroutine for inference
    """

    with open(f"{output_path}/rmse.txt", "w") as f:
        if not no_ref:
            f.write(f"Energy rmse (eV/atom): {rmse_dct['per_atom_energy']}\n")
            f.write(f"Force rmse (eV/A): {rmse_dct['force']}\n")
            if "stress" in rmse_dct:
                f.write(f"Stress rmse (kbar): {rmse_dct['stress']}\n")
        else:
            f.write("There is no ref data. RMSE is meaningless\n")
        for idx, info in enumerate(pred_ref_dct["info"]):
            f.write(f"{idx}: {info}\n")

    with open(f"{output_path}/energies.csv", "w", newline="") as f:
        fieldnames = ["StructureID", "Pred_energy(eV/atom)", "Ref_energy(eV/atom)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (pred, ref) in enumerate(zip(pred_ref_dct["per_atom_energy"][0],
                                            pred_ref_dct["per_atom_energy"][1])):
            writer.writerow({"StructureID": i,
                             "Pred_energy(eV/atom)": pred,
                             "Ref_energy(eV/atom)": ref})

    with open(f"{output_path}/forces.csv", "w", newline="") as f:
        fieldnames = ["StructureID", "AtomIndex",
                      "Pred_force_x", "Pred_force_y", "Pred_force_z",
                      "Ref_force_x", "Ref_force_y", "Ref_force_z"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (pred, ref) in enumerate(zip(pred_ref_dct["force"][0],
                                            pred_ref_dct["force"][1])):
            for j, (pred_f, ref_f) in enumerate(zip(pred, ref)):
                writer.writerow({"StructureID": i,
                                 "AtomIndex": j,
                                 "Pred_force_x": pred_f[0],
                                 "Pred_force_y": pred_f[1],
                                 "Pred_force_z": pred_f[2],
                                 "Ref_force_x": ref_f[0],
                                 "Ref_force_y": ref_f[1],
                                 "Ref_force_z": ref_f[2]})
    if not "stress" in rmse_dct:
        return

    with open(f"{output_path}/stresses.csv", "w", newline="") as f:
        fieldnames = ["StructureID",
                      "Pred_stress_xx", "Pred_stress_yy", "Pred_stress_zz",
                      "Pred_stress_xy", "Pred_stress_yz", "Pred_stress_zx",
                      "Ref_stress_xx", "Ref_stress_yy", "Ref_stress_zz",
                      "Ref_stress_xy", "Ref_stress_yz", "Ref_stress_zx"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (pred, ref) in enumerate(zip(pred_ref_dct["stress"][0],
                                            pred_ref_dct["stress"][1])):
            writer.writerow({"StructureID": i,
                             "Pred_stress_xx": pred[0],
                             "Pred_stress_yy": pred[1],
                             "Pred_stress_zz": pred[2],
                             "Pred_stress_xy": pred[3],
                             "Pred_stress_yz": pred[4],
                             "Pred_stress_zx": pred[5],
                             "Ref_stress_xx": ref[0],
                             "Ref_stress_yy": ref[1],
                             "Ref_stress_zz": ref[2],
                             "Ref_stress_xy": ref[3],
                             "Ref_stress_yz": ref[4],
                             "Ref_stress_zx": ref[5]})
    return


def _postprocess_output_list(output_list):
    """
    subroutine for inference
    return:
        pred_ref_dct: dict of tuple(pred, ref) of list of numpy, energy and stress
                      has extra dimension
        pred_ref_concat_dct: dict of tuple of concatenated pred and ref tensors
        rmse_dct: dict of rmse
    """
    TO_KB = 1602.1766208  # eV/A^3 to kbar
    POSTPROCESS_KEY_DCT = \
        {"per_atom_energy":
            {"pred_key": KEY.PRED_PER_ATOM_ENERGY,
             "ref_key": KEY.PER_ATOM_ENERGY,
             "vdim": 1,
             "coeff": 1,
             },
         "force":
            {"pred_key": KEY.PRED_FORCE,
             "ref_key": KEY.FORCE,
             "vdim": 3,
             "coeff": 1,
             },
         "stress":
            {"pred_key": KEY.PRED_STRESS,
             "ref_key": KEY.STRESS,
             "vdim": 6,
             "coeff": TO_KB,
             }
         }

    def get_vector_component_and_rmse(pred_V, ref_V, vdim: int):
        pred_V_component = torch.reshape(pred_V, (-1,))
        ref_V_component = torch.reshape(ref_V, (-1,))
        mse = (pred_V_component - ref_V_component) ** 2
        mse = torch.reshape(mse, (-1, vdim))
        mses = mse.sum(dim=1).tolist()
        rmse = np.sqrt(np.mean(mses))
        return pred_V_component, ref_V_component, rmse

    rmse_dct = {"per_atom_energy": {}, "force": {}}
    if KEY.PRED_STRESS in output_list[0]:
        rmse_dct["stress"] = {}

    pred_ref_dct = {}
    pred_ref_concat_dct = {}
    nstct = len(output_list)
    for key in rmse_dct.keys():
        pks = POSTPROCESS_KEY_DCT[key]
        vdim = pks["vdim"]
        # as a list of tensors for rmse calc
        pred_list = [o[pks["pred_key"]] * pks["coeff"] for o in output_list]
        ref_list = [o[pks["ref_key"]] * pks["coeff"] for o in output_list]
        pred_tensor = torch.cat([t.view(-1, vdim) for t in pred_list], dim=0)
        ref_tensor = torch.cat([t.view(-1, vdim) for t in ref_list], dim=0)

        # restuct to numpy, it gives extra dimension to energy and stress but
        # there is no option.. (since force has num of atoms dimension)
        pred_list = [torch.squeeze(t).detach().numpy() for t in pred_list]
        ref_list = [torch.squeeze(t).detach().numpy() for t in ref_list]
        pred_ref_dct[key] = (pred_list, ref_list)
        pred_ref_concat_dct[key] = (pred_tensor, ref_tensor)
        _, _, rmse = get_vector_component_and_rmse(pred_tensor, ref_tensor, vdim)
        rmse_dct[key] = rmse
    pred_ref_dct["info"] = [o[KEY.INFO] for o in output_list]
    return pred_ref_dct, pred_ref_concat_dct, rmse_dct


def inference_main(checkpoint, fnames, output_path, num_cores=1, device="cpu"):
    checkpoint = torch.load(checkpoint, map_location=device)
    config = checkpoint["config"]
    cutoff = config[KEY.CUTOFF]
    type_map = config[KEY.TYPE_MAP]

    model = load_model_from_checkpoint(checkpoint)
    model.set_is_batch_data(False)
    model.eval()

    head = os.path.basename(fnames[0])  # expect user did right thing
    atoms_list = None
    no_ref = False
    if head.endswith('sevenn_data'):
        inference_set = load_sevenn_datas(fnames, cutoff, type_map)
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
    else:
        inference_set.toggle_requires_grad_of_data(KEY.EDGE_VEC, True)

    infer_list = inference_set.to_list()
    output_list = []

    model.to(device)
    # TODO: make it multicore parallel (you know it is not pickable directly...)
    for datum in tqdm(infer_list):
        datum.to(device)
        output = model(datum)
        for k, v in output:
            if isinstance(v, torch.Tensor):
                output[k] = v.detach().cpu()
            else:
                output[k] = v
        output_list.append(output)

    pred_ref_dct, pred_ref_concat_dct, rmse_dct \
        = _postprocess_output_list(output_list)

    _write_inference_csv(pred_ref_dct, rmse_dct, output_path, no_ref=no_ref)


if __name__ == "__main__":
    prefix = "/home/parkyutack/SEVENNet/example_inputs/testing/"
    checkpoint = f"{prefix}/checkpoint_5.pth"
    train = f"{prefix}/train.sevenn_data"
    valid = f"{prefix}/valid.sevenn_data"
    outcars = f"{prefix}/outcars/OUTCAR*"
    outcars = f"{prefix}/tmp/OUTCAR*"
    poscars = f"{prefix}/poscars/POSCAR*"
    #poscars = f"{prefix}/tmp/POSCAR*"

    #inference_main(checkpoint, outcars, None, num_cores=1, device="cpu")
    #inference_main(checkpoint, train, "debug", num_cores=1, device="cpu")
    inference_main(checkpoint, poscars, "debug", num_cores=1, device="cpu")


