import copy
import csv
import glob
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime

import ase
import ase.calculators as calculators
import ase.calculators.lammps as ase_lammps
import ase.io
import ase.io.lammpsdata
import numpy as np
import torch

import sevenn._keys as KEY
from sevenn._const import SEVENN_VERSION

"""
Tests for SevenNet

2023-04-14
"""
test_root = os.path.dirname(os.path.abspath(__file__))

INPUT_YAML = sys.argv[1] if __file__ == 'testbot.py' else None
OUTCAR = f'{test_root}/OUTCAR_test'
LMP_BIN = 'lmp'

LMP_SCRIPT = f'{test_root}/LMP_SCRIPT/oneshot.lmp'

ATOL = 1e-6
RTOL = 1e-5

e3gnn_found = False
e3gnn_parallel_found = False


def bar():
    print('-' * 50)


def subprocess_routine(cmd, name, print_stdout=False):
    start = datetime.now()
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        print(f'return code: {res.returncode}')
        print(res.stderr.decode('utf-8'))
        raise RuntimeError(f'{name} failed')
    else:
        print(f'{name} succeeded')
        elapsed = str(datetime.now() - start)
        print(f'{name} wall time:', elapsed)

    if print_stdout:
        print(res.stdout.decode('utf-8'))


def greeting():
    global e3gnn_found, e3gnn_parallel_found
    global RTOL, ATOL
    print(f'SevenNet version {SEVENN_VERSION}')
    print(f'PyTorch version {torch.__version__}')
    print(f'ASE version {ase.__version__}')

    try:
        re = subprocess.run([LMP_BIN, '-h'], capture_output=True)
        stdout = re.stdout.decode('utf-8')
        lines = stdout.split('\n')

        kwords = ['Large-scale', 'MPI', 'Compiler', 'OS:', 'C++']

        print('\nCompiled lammps metadata:')
        for ln in lines:
            if any([kw in ln for kw in kwords]):
                print(ln)
            if 'e3gnn' in ln:
                e3gnn_found = True
            if 'e3gnn/parallel' in ln:
                e3gnn_parallel_found = True

        if not e3gnn_found:
            print('e3gnn not found, lammps serial test will be skipped.')
            print('Did you correctly put e3gnn.cpp/h in lammps/src?')
        if not e3gnn_parallel_found:
            print(
                'e3gnn_parallel not found, lammps parallel test will be'
                ' skipped.'
            )
            print('Did you correctly put e3gnn_parallel.cpp/h in lammps/src?')
            print("Also, don't forget to put comm_brick.cpp/h in lammps/src")

    except Exception as e:
        print(e)
        print('Exception while executing LAMMPS, skip related tests')
        e3gnn_found = False
        e3gnn_parallel_found = False

    if torch.cuda.is_available():
        print('CUDA available, we lower accuracy constraint and use the GPU')
        ATOL *= 10
        RTOL *= 10
    else:
        print('Use CPU for the test')


def atoms_info_print(atoms):
    print('Atoms info:')
    print(f'Number of atoms: {len(atoms)}')
    species = Counter(atoms.get_chemical_symbols())
    print(f'Species: {species}')
    print(f'Cell:\n {atoms.get_cell().array}')
    print(f'Energy: {atoms.get_potential_energy(force_consistent=True)}')
    print(
        'Energy per atom:'
        f' {atoms.get_potential_energy(force_consistent=True) / len(atoms)}'
    )
    print(f'Position of first atom: {atoms.get_positions()[0]}')
    print(f'Force of first atom: {atoms.get_forces()[0]}')
    try:
        stress = atoms.get_stress(voigt=False)
        stress_kB = stress * 1602.1766208
        print(f'Stress(kB):\n {stress_kB}')
    except:
        print('Stress failed to load')


def graph_build_test():
    print('sevenn_graph_build test')

    graph_build_cmd = (
        f'sevenn_graph_build {OUTCAR} 4.0 -f vasp-out -o data'.split()
    )
    subprocess_routine(graph_build_cmd, 'sevenn_graph_build')


def sevenn_train_test():
    print('sevenn training test')
    ori_dir = os.getcwd()

    os.makedirs('./training', exist_ok=True)
    shutil.copy('./data.sevenn_data', './training/data.sevenn_data')
    shutil.copy(INPUT_YAML, './training/input.yaml')

    os.chdir('./training')
    sevenn_train_cmd = f'sevenn input.yaml'.split()

    subprocess_routine(sevenn_train_cmd, 'sevenn_train')

    with open('./log.sevenn', 'r') as f:
        lines = f.readlines()

    metrics = ['Energy_RMSE', 'Force_RMSE']
    rmse = None
    for ln in lines:
        if ln.startswith('Valid'):
            rmse = ln.split()[1:3]  # TODO: more robust way
    rmse_dct = {k: float(v) for k, v in zip(metrics, rmse)}
    cp_path = os.path.abspath('./checkpoint_best.pth')

    os.chdir(ori_dir)
    return rmse_dct, cp_path


def sevenn_infer_to_atoms(inference_folder):
    def np_str_to_np_array(np_str):
        tmp = np_str.replace('[', '').replace(']', '').split()
        return np.array([float(x) for x in tmp])

    info_list = None
    if os.path.exists(f'{inference_folder}/info.csv'):
        with open(f'{inference_folder}/info.csv', 'r') as f:
            reader = csv.DictReader(f)
            info_list = [row for row in reader]

    with open(f'{inference_folder}/per_graph.csv', 'r') as f:
        reader = csv.DictReader(f)
        per_graph = [row for row in reader]

    if info_list is not None:
        per_graph = [
            {**row, KEY.INFO: info_list[i]} for i, row in enumerate(per_graph)
        ]

    with open(f'{inference_folder}/per_atom.csv', 'r') as f:
        reader = csv.DictReader(f)
        per_atom = [row for row in reader]

    atoms_list = []
    for _, row in enumerate(per_graph):
        info = row[KEY.INFO]
        natoms = int(row[KEY.NUM_ATOMS])
        cell_a = np_str_to_np_array(row[KEY.CELL + '_a'])
        cell_b = np_str_to_np_array(row[KEY.CELL + '_b'])
        cell_c = np_str_to_np_array(row[KEY.CELL + '_c'])
        cell = np.array([cell_a, cell_b, cell_c])
        calc_res = {}
        energy = float(row[KEY.PRED_TOTAL_ENERGY])
        calc_res['energy'] = energy
        calc_res['free_energy'] = energy
        if (
            KEY.PRED_STRESS + '_xx' in row
            and row[KEY.PRED_STRESS + '_xx'] != '-'
        ):
            stress_xx = float(row[KEY.PRED_STRESS + '_xx'])
            stress_yy = float(row[KEY.PRED_STRESS + '_yy'])
            stress_zz = float(row[KEY.PRED_STRESS + '_zz'])
            stress_xy = float(row[KEY.PRED_STRESS + '_xy'])
            stress_zx = float(row[KEY.PRED_STRESS + '_zx'])
            stress_yz = float(row[KEY.PRED_STRESS + '_yz'])
            stress = np.array([
                stress_xx,
                stress_yy,
                stress_zz,
                stress_yz,
                stress_zx,
                stress_xy,
            ])
            stress = -1 * stress / 1602.1766208  # convert kB to eV/A^3
            calc_res['stress'] = stress
        per_atom_data = per_atom[:natoms]
        per_atom = per_atom[natoms:]
        species = []
        pos = []
        forces = []
        for _, frow in enumerate(per_atom_data):
            species.append(frow[KEY.ATOMIC_NUMBERS])
            pos.append([
                float(frow[KEY.POS + '_x']),
                float(frow[KEY.POS + '_y']),
                float(frow[KEY.POS + '_z']),
            ])
            forces.append([
                float(frow[KEY.PRED_FORCE + '_x']),
                float(frow[KEY.PRED_FORCE + '_y']),
                float(frow[KEY.PRED_FORCE + '_z']),
            ])
        species = np.array(species)
        pos = np.array(pos)
        forces = np.array(forces)
        calc_res['forces'] = forces
        atoms = ase.Atoms(
            numbers=species, positions=pos, cell=cell, pbc=True, info=info
        )
        calculator = calculators.singlepoint.SinglePointCalculator(
            atoms, **calc_res
        )
        atoms = calculator.get_atoms()
        atoms_list.append(atoms)

    return atoms_list


def sevenn_inferece_test(cp_path: str):
    print('sevenn_inference test')
    sevenn_inference_cmd = (
        f'sevenn_inference -o inference {cp_path} {OUTCAR}'.split()
    )

    subprocess_routine(sevenn_inference_cmd, 'sevenn_inference')

    with open('./inference/rmse.txt', 'r') as f:
        lines = f.readlines()
    delm = 2  # energy and force (stress RMSE might missing)
    rmse_dct = {l.split()[0]: float(l.split()[-1]) for l in lines[:delm]}

    infer_atoms = sevenn_infer_to_atoms('./inference')

    # TODO: for now, assume there is only one structure to test
    return rmse_dct, infer_atoms[0]


def sevenn_get_model_test(cp_path, is_parallel):
    print('sevenn get model')
    os.makedirs('./potential', exist_ok=True)
    name = 'sevenn_get_model'
    if is_parallel:
        sevenn_get_model_cmd = (
            f'sevenn_get_model -p {cp_path} -o potential/parallel'.split()
        )
        name += ' parallel'
    else:
        sevenn_get_model_cmd = (
            f'sevenn_get_model {cp_path} -o potential/serial'.split()
        )
        name += ' serial'
    subprocess_routine(sevenn_get_model_cmd, name)
    if is_parallel:
        pts = sorted(glob.glob('./potential/parallel*.pt'))
        lmp_pot_str = f'{len(pts)}'
        for pt in pts:
            lmp_pot_str += f' {os.path.abspath(pt)}'
    else:
        lmp_pot_str = os.path.abspath('./potential/serial.pt')
    return lmp_pot_str


def sevenn_lmp_test(
    atoms,
    pair_style,
    pot_str,
    test_dir,
    mpicmd=None,
    ncores=None,
    replicate='1 1 1',
    rot_atoms_back=True,
):
    def lammps_results_to_atoms(lammps_log, force_dump, read_stress=True):
        # read lammps log file
        # TODO: May not robust, use lammps python
        with open(lammps_log, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if not line.startswith('Per MPI rank memory allocation'):
                continue
            lmp_log = {
                k: eval(v)
                for k, v in zip(lines[i + 1].split(), lines[i + 2].split())
            }
            break

        assert 'PotEng' in lmp_log

        # read lammps dump file
        latoms = ase.io.read(force_dump, format='lammps-dump-text', index=':')[
            0
        ]
        latoms.calc.results['energy'] = lmp_log['PotEng']
        latoms.calc.results['free_energy'] = lmp_log['PotEng']
        latoms.info = {
            'data_from': 'lammps',
            'lmp_log': lmp_log,
            'lmp_dump': force_dump,
        }

        if read_stress:
            stress = np.array([
                [lmp_log['Pxx'], lmp_log['Pxy'], lmp_log['Pxz']],
                [lmp_log['Pxy'], lmp_log['Pyy'], lmp_log['Pyz']],
                [lmp_log['Pxz'], lmp_log['Pyz'], lmp_log['Pzz']],
            ])
            stress = (
                -1 * stress / 1602.1766208 / 1000
            )  # convert bars to eV/A^3
            latoms.calc.results['stress'] = stress

        # TODO: for now, parse only first frame, maybe we could parse all frames
        #       by sync log and force dump (Steps in log & 'index' in read)

        return latoms

    ori_dir = os.getcwd()
    os.makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)

    chem = list(set(atoms.get_chemical_symbols()))

    # Way to ase handle lammps structure
    prism = ase_lammps.coordinatetransform.Prism(atoms.get_cell())
    ase.io.lammpsdata.write_lammps_data(
        './lammps_stct', atoms, prismobj=prism, specorder=chem
    )

    with open(LMP_SCRIPT, 'r') as f:
        cont = f.read()

    # Using -var switch is not work since it includes space
    var_dct = {}
    var_dct['__ELEMENT__'] = ' '.join(chem)
    var_dct['__LMP_STCT__'] = os.path.abspath('./lammps_stct')
    var_dct['__POTENTIALS__'] = pot_str
    var_dct['__PAIR_STYLE__'] = pair_style
    var_dct['__REPLICATE__'] = replicate
    for key, val in var_dct.items():
        cont = cont.replace(key, val)

    with open('./in.lmp', 'w') as f:
        f.write(cont)

    lmp_cmd = f'{LMP_BIN} -in in.lmp'
    if mpicmd is not None:
        lmp_cmd = f'{mpicmd} -np {ncores} {lmp_cmd}'
    lmp_cmd = lmp_cmd.split()

    print('LAMMPS test')
    subprocess_routine(lmp_cmd, 'lammps run')

    # TODO: after implement stress for parallel, update this
    read_stress = True
    if pair_style == 'e3gnn/parallel':
        read_stress = False
    latoms = lammps_results_to_atoms(
        './log.lammps', './force.dump', read_stress
    )

    # To obey lammps convention ase rotated cell and we just read that cell
    # by ase, cell & positions are rotate via cell @ rot_mat, pos dot rot_mat
    if rot_atoms_back:
        rot_mat = prism.rot_mat
        results = copy.deepcopy(latoms.calc.results)
        r_force = np.dot(results['forces'], rot_mat.T)
        results['forces'] = r_force
        if 'stress' in results:
            # see ase.calculators.lammpsrun.py
            stress_tensor = results['stress']
            stress_atoms = np.dot(np.dot(rot_mat, stress_tensor), rot_mat.T)
            results['stress'] = stress_atoms
        r_cell = latoms.get_cell() @ rot_mat.T
        latoms.set_cell(r_cell, scale_atoms=True)
        latoms = calculators.singlepoint.SinglePointCalculator(
            latoms, **results
        ).get_atoms()

    os.chdir(ori_dir)

    return latoms


def compare_atoms(atoms1, atoms2, rtol=1e-5, atol=1e-8):
    if not isinstance(atoms1, ase.Atoms):
        raise TypeError(
            f'atoms1 should be instance of ase.Atoms, not {type(atoms1)}'
        )
    if not isinstance(atoms2, ase.Atoms):
        raise TypeError(
            f'atoms2 should be instance of ase.Atoms, not {type(atoms2)}'
        )

    if atoms1.get_chemical_symbols() != atoms2.get_chemical_symbols():
        raise ValueError('atoms1 and atoms2 have different chemical symbols')

    if not np.allclose(
        atoms1.get_cell(), atoms2.get_cell(), rtol=rtol, atol=atol
    ):
        print(atoms1.get_cell())
        print(atoms2.get_cell())
        raise ValueError('atoms1 and atoms2 have different cell')

    """
    if not atoms1.calc or not atoms2.calc:
        raise ValueError('atoms1 and atoms2 should have calculator')

    if not isinstance(atoms1.calc, SinglePointCalculator):
        raise TypeError(
            'atoms1.calc should be instance of'
            f' ase.calculator.SinglePointCalculator, not {type(atoms1.calc)}'
        )
    if not isinstance(atoms2.calc, SinglePointCalculator):
        raise TypeError(
            'atoms2.calc should be instance of'
            f' ase.calculator.SinglePointCalculator, not {type(atoms2.calc)}'
        )
    """

    if not np.allclose(
        atoms1.get_potential_energy(),
        atoms2.get_potential_energy(),
        rtol=rtol,
        atol=atol,
    ):
        print('atoms1 energy:', atoms1.get_potential_energy())
        print('atoms2 energy:', atoms2.get_potential_energy())
        raise ValueError('atoms1 and atoms2 have different energy')

    if not np.allclose(
        atoms1.get_forces(),
        atoms2.get_forces(),
        rtol=rtol,
        atol=atol,
    ):
        # increasing atol iteratively to find the difference
        f1 = atoms1.get_forces()
        f2 = atoms2.get_forces()
        flag = False
        for _ in range(3):
            atol *= 10
            if np.allclose(
                f1,
                f2,
                rtol=rtol,
                atol=atol,
            ):
                flag = True
                break
        if not flag:
            raise ValueError('atoms1 and atoms2 have different forces')
        else:
            print(f'Warning: force comparison failed, but within atol={atol}')

    try:
        s1 = atoms1.get_stress(voigt=False)
        s2 = atoms2.get_stress(voigt=False)
        if not np.allclose(s1, s2, rtol=rtol, atol=atol):
            print('atoms1 stress:', s1)
            print('atoms2 stress:', s2)
            raise ValueError('atoms1 and atoms2 have different stress')
    except calculators.calculator.PropertyNotImplementedError:
        print('One of atoms does not have stress, skip stress comparison')

    return True


def train_infer_rmse_test(train_rmse_dct, infer_rmse_dct, atol=1e-6):
    if np.isclose(
        train_rmse_dct['Energy_RMSE'], infer_rmse_dct['Energy'], atol=atol
    ):
        print('Energy RMSE test passed')
    else:
        raise ValueError(
            f"Energy RMSE test failed: {train_rmse_dct['Energy_RMSE']} !="
            f" {infer_rmse_dct['Energy']}"
        )

    if np.isclose(
        train_rmse_dct['Force_RMSE'], infer_rmse_dct['Force'], atol=atol
    ):
        print('Force RMSE test passed')
    else:
        raise ValueError(
            f"Force RMSE test failed: {train_rmse_dct['Force_RMSE']} !="
            f" {infer_rmse_dct['Force']}"
        )

    return True


def sevenn_calculator_test(checkpoint, atoms, device):
    from sevenn.sevennet_calculator import SevenNetCalculator

    cal = SevenNetCalculator(checkpoint, device='cpu')
    ref_atoms = copy.deepcopy(atoms)
    atoms.set_calculator(cal)

    if compare_atoms(ref_atoms, atoms, rtol=RTOL, atol=ATOL):
        print('SevenNetCalculator test passed')


def main():
    if __file__ != 'testbot.py':
        raise RuntimeError('Please execute the script in the same directory')

    print('Check necessary files')
    for check in [OUTCAR, LMP_SCRIPT, INPUT_YAML]:
        if not os.path.exists(check):
            raise FileNotFoundError(f'{check} not found')

    greeting()
    bar()

    atoms = ase.io.read(OUTCAR, format='vasp-out')
    atoms_info_print(atoms)
    bar()

    graph_build_test()
    bar()

    train_rmse_dct, cp_path = sevenn_train_test()
    cp_path = './training/checkpoint_best.pth'
    bar()

    infer_rmse_dct, infer_atoms = sevenn_inferece_test(cp_path)
    bar()

    train_infer_rmse_test(train_rmse_dct, infer_rmse_dct, atol=ATOL)
    bar()

    sevenn_calculator_test(cp_path, infer_atoms, device='auto')
    bar()

    if e3gnn_found:
        seri_pot_str = sevenn_get_model_test(cp_path, False)
        bar()
        serial_atoms = sevenn_lmp_test(
            atoms, 'e3gnn', seri_pot_str, './e3gnn_test'
        )
        bar()

        if compare_atoms(infer_atoms, serial_atoms, rtol=RTOL, atol=ATOL):
            print('LAMMPS e3gnn serial test passed')
        bar()
    else:
        print('e3gnn not found, lammps serial test skipped')

    if e3gnn_found and e3gnn_parallel_found:
        para_pot_str = sevenn_get_model_test(cp_path, True)
        bar()

        print('Replicate cell by 2x1x1 for parallel test')
        serial_211 = sevenn_lmp_test(
            atoms, 'e3gnn', seri_pot_str, './e3gnn_211', replicate='2 1 1'
        )
        bar()
        para_211 = sevenn_lmp_test(
            atoms,
            'e3gnn/parallel',
            para_pot_str,
            './e3gnn_parallel_211',
            mpicmd='mpirun',
            replicate='2 1 1',
            ncores=2,
        )
        bar()
        if compare_atoms(serial_211, para_211, rtol=RTOL, atol=ATOL):
            print('Serial and parallel test passed')
        bar()
    else:
        print('e3gnn_parallel not found, lammps parallel test skipped')

    print('Energy RMSE (as classifier btw tests):')
    print(f"{infer_rmse_dct['Energy']}")


if __name__ == '__main__':
    main()
