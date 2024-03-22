import copy
import glob
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime

import ase
import ase.io
import numpy as np
import torch
from ase.calculators.lammps.coordinatetransform import Prism
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.lammpsdata import write_lammps_data

import sevenn._keys as KEY
from sevenn._const import SEVENN_VERSION

"""
End to end test for the whole package.

2023-10-08
"""

OUTCAR = sys.argv[1]
INPUT_YAML = sys.argv[2]
LMP_BIN = 'lmp'

# These code are strongly dependent with below two files
LMP_SCRIPT = os.path.abspath('./LMP_SCRIPT/oneshot.lmp')  # pre-defined

e3gnn_found = False
e3gnn_parallel_found = False


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
    print('Hello, world!')
    print(f'SEVENNet version {SEVENN_VERSION}')
    print(f'PyTorch version {torch.__version__}')
    print(f'ASE version {ase.__version__}')

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
            'e3gnn_parallel not found, lammps parallel test will be skipped.'
        )
        print('Did you correctly put e3gnn_parallel.cpp/h in lammps/src?')
        print("Also, don't forget to put comm_brick.cpp/h in lammps/src")


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
    print('sevenn_graph_build on OUTCAR')

    graph_build_cmd = (
        f'sevenn_graph_build {OUTCAR} 4.0 -f vasp-out -o data'.split()
    )
    subprocess_routine(graph_build_cmd, 'sevenn_graph_build')


def sevenn_train_test():
    print('\nsevenn_train on data')
    ori_dir = os.getcwd()

    os.makedirs('./training', exist_ok=True)
    shutil.copy('./data.sevenn_data', './training/data.sevenn_data')
    shutil.copy(f'{INPUT_YAML}', './training/input.yaml')

    os.chdir('./training')
    sevenn_train_cmd = f'sevenn input.yaml'.split()

    subprocess_routine(sevenn_train_cmd, 'sevenn_train')

    with open('./log.sevenn', 'r') as f:
        lines = f.readlines()

    metrics = ['EnergyRMSE', 'ForceRMSE']
    rmse = None
    for ln in lines:
        if ln.startswith('Valid'):
            rmse = ln.split()[1:3]
    rmse_dct = {k: float(v) for k, v in zip(metrics, rmse)}
    cp_path = os.path.abspath('./checkpoint_best.pth')

    os.chdir(ori_dir)
    return rmse_dct, cp_path


def sevenn_inferece_test(cp_path: str):
    def sevenn_infer_to_atoms(inference_folder):
        import csv
        import os

        import ase.atoms
        from ase.calculators.singlepoint import SinglePointCalculator

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
                {**row, KEY.INFO: info_list[i]}
                for i, row in enumerate(per_graph)
            ]

        with open(f'{inference_folder}/per_atom.csv', 'r') as f:
            reader = csv.DictReader(f)
            per_atom = [row for row in reader]

        atoms_list = []
        for i, row in enumerate(per_graph):
            info = row[KEY.INFO]
            label = row[KEY.USER_LABEL]
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
            for j, frow in enumerate(per_atom_data):
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
            atoms = ase.atoms.Atoms(
                numbers=species, positions=pos, cell=cell, pbc=True, info=info
            )
            calculator = SinglePointCalculator(atoms, **calc_res)
            atoms = calculator.get_atoms()
            atoms_list.append(atoms)

        return atoms_list

    print('\nsevenn_inference on OUTCAR')
    sevenn_inference_cmd = (
        f'sevenn_inference -o inference {cp_path} {OUTCAR}'.split()
    )

    subprocess_routine(sevenn_inference_cmd, 'sevenn_inference')

    with open('./inference/rmse.txt', 'r') as f:
        lines = f.readlines()
    is_stress = os.path.exists('./inference/stress.csv')
    delm = 3 if is_stress else 2
    rmse_dct = {l.split()[0]: float(l.split()[-1]) for l in lines[:delm]}

    infer_atoms = sevenn_infer_to_atoms('./inference')

    # TODO: for now, assume there is only one structure to test
    return rmse_dct, infer_atoms[0]


def sevenn_get_model_test(cp_path, is_parallel):
    print('\nsevenn_get_model')
    os.makedirs('./potential', exist_ok=True)
    if is_parallel:
        sevenn_get_model_cmd = (
            f'sevenn_get_model -p {cp_path} -o potential/parallel'.split()
        )
    else:
        sevenn_get_model_cmd = (
            f'sevenn_get_model {cp_path} -o potential/serial'.split()
        )

    subprocess_routine(sevenn_get_model_cmd, 'sevenn_get_model')

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
    def lammps_results_to_atoms(lammps_log, force_dump):
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.io import lammpsdata, read

        # read lammps log file
        # TODO: Read stress from lammps after stress in lmp is implemented
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
        latoms = read(force_dump, format='lammps-dump-text', index=':')[0]
        latoms.calc.results['energy'] = lmp_log['PotEng']
        latoms.calc.results['free_energy'] = lmp_log['PotEng']
        latoms.info = {
            'data_from': 'lammps',
            'lmp_log': lmp_log,
            'lmp_dump': force_dump,
        }

        # TODO: for now, parse only first frame, maybe we could parse all frames
        #       by sync log and force dump (Steps in log & 'index' in read)

        return latoms

    ori_dir = os.getcwd()
    os.makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)

    chem = list(set(atoms.get_chemical_symbols()))

    # Way to ase handle lammps structure
    prism = Prism(atoms.get_cell())
    write_lammps_data('./lammps_stct', atoms, prismobj=prism, specorder=chem)

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

    print('\nLAMMPS test')
    subprocess_routine(lmp_cmd, 'lammps run')

    latoms = lammps_results_to_atoms('./log.lammps', './force.dump')

    # To obey lammps convention ase rotated cell and we just read that cell
    # by ase, cell & positions are rotate via cell @ rot_mat, pos dot rot_mat
    if rot_atoms_back:
        rot_mat = prism.rot_mat
        results = copy.deepcopy(latoms.calc.results)
        r_force = np.dot(results['forces'], rot_mat.T)
        results['forces'] = r_force
        r_cell = latoms.get_cell() @ rot_mat.T
        latoms.set_cell(r_cell, scale_atoms=True)

        latoms = SinglePointCalculator(latoms, **results).get_atoms()

    os.chdir(ori_dir)

    return latoms


def compare_atoms(atoms1, atoms2, rtol=1e-5, atol=1e-8):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    if not isinstance(atoms1, Atoms):
        raise TypeError(
            f'atoms1 should be instance of ase.Atoms, not {type(atoms1)}'
        )
    if not isinstance(atoms2, Atoms):
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

    if not np.allclose(
        atoms1.calc.results['energy'],
        atoms2.calc.results['energy'],
        rtol=rtol,
        atol=atol,
    ):
        print('atoms1 energy:', atoms1.calc.results['energy'])
        print('atoms2 energy:', atoms2.calc.results['energy'])
        raise ValueError('atoms1 and atoms2 have different energy')
    print(f'atoms1 and atoms2 have same total energy')

    if not np.allclose(
        atoms1.calc.results['forces'],
        atoms2.calc.results['forces'],
        rtol=rtol,
        atol=atol,
    ):
        # increasing atol iteratively to find the difference
        flag = False
        for i in range(4):
            atol *= 10
            if np.allclose(
                atoms1.calc.results['forces'],
                atoms2.calc.results['forces'],
                rtol=rtol,
                atol=atol,
            ):
                flag = True
                break
        if not flag:
            raise ValueError('atoms1 and atoms2 have truely different forces')
        else:
            print(f'atoms1 and atoms2 have same forces, within atol={atol}')

    return True


def train_infer_rmse_test(train_rmse_dct, infer_rmse_dct):
    print('Since we used validation set of training to inference, we expect')
    print(
        'validation rmse of sevenn_train and rmse of sevenn_infer to be same'
    )

    if np.isclose(
        train_rmse_dct['EnergyRMSE'], infer_rmse_dct['Energy'], atol=1e-6
    ):
        print('Energy RMSE test passed')
    else:
        raise ValueError(
            f"Energy RMSE test failed: {train_rmse_dct['EnergyRMSE']} !="
            f" {infer_rmse_dct['Energy']}"
        )

    if np.isclose(
        train_rmse_dct['ForceRMSE'], infer_rmse_dct['Force'], atol=1e-6
    ):
        print('Force RMSE test passed')
    else:
        raise ValueError(
            f"Force RMSE test failed: {train_rmse_dct['ForceRMSE']} !="
            f" {infer_rmse_dct['Force']}"
        )

    return True


def sevenn_calculator_test(atoms):
    from sevenn.sevennet_calculator import SevenNetCalculator
    if os.getenv("SEVENNET_0_CP") is None:
        print("SEVENNET_0_CP not set")
        return
    cal = SevenNetCalculator(device="cpu")
    atoms.set_calculator(cal)
    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_stress()


def main():
    if __file__ != 'testbot.py':
        print('Change directory to the location of testbot.py')
        os.chdir(os.path.dirname(__file__))

    print('Check files for testbot exist')
    for check in [OUTCAR, LMP_SCRIPT, INPUT_YAML]:
        if not os.path.exists(check):
            raise FileNotFoundError(f'{check} not found')

    greeting()
    print('----------------------------------------------------')

    atoms = ase.io.read(OUTCAR, format='vasp-out')
    atoms_info_print(atoms)
    print('----------------------------------------------------')

    graph_build_test()
    print('----------------------------------------------------')

    train_rmse_dct, cp_path = sevenn_train_test()
    cp_path = './training/checkpoint_best.pth'
    print('----------------------------------------------------')

    infer_rmse_dct, infer_atoms = sevenn_inferece_test(cp_path)
    print('----------------------------------------------------')

    sevenn_calculator_test(atoms)
    print('----------------------------------------------------')

    train_infer_rmse_test(train_rmse_dct, infer_rmse_dct)
    print('----------------------------------------------------')

    if e3gnn_found:
        print('\ntest lmp serial potential')
        seri_pot_str = sevenn_get_model_test(cp_path, False)
        print('----------------------------------------------------')
        serial_atoms = sevenn_lmp_test(
            atoms, 'e3gnn', seri_pot_str, './e3gnn_test'
        )
        print('----------------------------------------------------')

        if compare_atoms(infer_atoms, serial_atoms):
            print('\nInference and serial test passed')
        print('----------------------------------------------------')

    if e3gnn_found and e3gnn_parallel_found:
        print('\ntest lmp parallel potential')
        para_pot_str = sevenn_get_model_test(cp_path, True)
        print('----------------------------------------------------')

        print('\nReplicate cell by 3x3x3 to get correct force for parallel')
        serial_333 = sevenn_lmp_test(
            atoms, 'e3gnn', seri_pot_str, './e3gnn_333', replicate='3 3 3'
        )
        print('----------------------------------------------------')
        print('\nSerial 333 lammps run done')
        para_333 = sevenn_lmp_test(
            atoms,
            'e3gnn/parallel',
            para_pot_str,
            './e3gnn_parallel_333',
            mpicmd='mpirun',
            replicate='3 3 3',
            ncores=2,
        )
        print('----------------------------------------------------')
        print('\nParallel 333 lammps run done')

        if compare_atoms(serial_333, para_333):
            print('\nSerial and parallel test passed')
        print('----------------------------------------------------')

    print('Energy RMSE (as classifier btw tests):')
    print(f"{infer_rmse_dct['Energy']}")


if __name__ == '__main__':
    main()
