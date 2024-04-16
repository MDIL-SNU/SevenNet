import glob
import os
import shutil
import sys

import yaml

# Add parent directory to sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
)
import tests.testbot as tbot

"""
Test for checkpoint sanity

2024-04-13
"""

TARGET = sys.argv[1] if len(sys.argv) > 1 else 'cp_sevennet_0'

LMP_SCRIPT = tbot.LMP_SCRIPT
LMP_BIN = tbot.LMP_BIN

if TARGET == 'cp_sevennet_0':
    print('Using SEVENNET_0_CP environment variable')
    cp_path = os.getenv('SEVENNET_0_CP')
else:
    cp_path = f'{TARGET}/checkpoint.pth'
CHECKPOINT = cp_path
CONTINUE_INPUT_YAML = f'{TARGET}/continue_input.yaml'
INFERENCE_REF_DIR = f'{TARGET}/inference_ref'
DATA_REF = f'{TARGET}/data_ref.sevenn_data'

CP_OUTCAR = None
CP_ATOL = None
CP_RTOL = None
CP_e3gnn_found = None
CP_e3gnn_parallel_found = None


def bar():
    print('-' * 50)


def sevenn_continue_test():
    print('sevenn continue test')
    ori_dir = os.getcwd()

    os.makedirs('continue_test', exist_ok=True)
    shutil.copy(CHECKPOINT, 'continue_test/checkpoint.pth')
    shutil.copy(CONTINUE_INPUT_YAML, 'continue_test/continue_input.yaml')

    os.chdir('continue_test')
    sevenn_train_cmd = 'sevenn continue_input.yaml'.split()

    with open('continue_input.yaml', 'r') as fstream:
        inputs = yaml.safe_load(fstream)
    train = inputs['train']
    train['epoch'] = 1
    train['error_record'] = [
        ['Energy', 'RMSE'],
        ['Force', 'RMSE'],
        ['Stress', 'RMSE'],
        ['TotalLoss', 'None'],
    ]
    train['per_epoch'] = 100
    train['continue']['checkpoint'] = 'checkpoint.pth'
    train['continue']['reset_optimize'] = False
    train['continue']['reset_scheduler'] = False

    inputs['data']['batch_size'] = 1
    inputs['data']['load_dataset_path'] = '../small.sevenn_data'
    inputs['data']['load_validset_path'] = '../small.sevenn_data'

    with open('continue_input.yaml', 'w') as fstream:
        yaml.safe_dump(inputs, fstream)

    tbot.subprocess_routine(sevenn_train_cmd, 'sevenn continue')

    with open('./log.sevenn', 'r') as f:
        lines = f.readlines()

    metrics = ['Energy_RMSE', 'Force_RMSE', 'Stress_RMSE']
    rmse = None
    for ln in lines:
        if ln.startswith('Valid'):
            rmse = ln.split()[1:]
    rmse_dct = {k: float(v) for k, v in zip(metrics, rmse)}
    os.chdir(ori_dir)

    return rmse_dct


def sevenn_inference_test_with_compare():
    print('sevenn_inference on reference dataset')
    sevenn_inference_cmd = (
        f'sevenn_inference -o inference {CHECKPOINT} {DATA_REF}'.split()
    )

    tbot.subprocess_routine(sevenn_inference_cmd, 'sevenn inference')
    infer_atoms = tbot.sevenn_infer_to_atoms('./inference')
    ref_atoms = tbot.sevenn_infer_to_atoms(INFERENCE_REF_DIR)
    assert len(infer_atoms) == len(ref_atoms)

    for i, (infer, ref) in enumerate(zip(infer_atoms, ref_atoms)):
        try:
            if tbot.compare_atoms(infer, ref, CP_ATOL, CP_RTOL):
                continue
        except ValueError:
            print(f'Test failed error in {i}th structure')
            raise
    print('Inference test passed')
    print('Prepare reference data for OUTCAR_test')
    sevenn_inference_cmd = (
        f'sevenn_inference -o inference_small {CHECKPOINT} {CP_OUTCAR}'.split()
    )
    tbot.subprocess_routine(sevenn_inference_cmd, 'sevenn inference small')
    ref_small_atoms = tbot.sevenn_infer_to_atoms('./inference_small')
    assert len(ref_small_atoms) == 1

    return ref_small_atoms[0]


def main():
    global CP_ATOL, CP_RTOL, CP_e3gnn_found, CP_e3gnn_parallel_found, CP_OUTCAR
    if __file__ != 'checkpoint_testbot.py':
        raise RuntimeError('Please execute the script in the same directory')

    print('Check necessary files')
    for check in [
        LMP_SCRIPT,
        CHECKPOINT,
        CONTINUE_INPUT_YAML,
        INFERENCE_REF_DIR,
        DATA_REF,
    ]:
        if not os.path.exists(check):
            raise FileNotFoundError(f'{check} not found')

    tbot.greeting()
    bar()

    CP_ATOL = tbot.ATOL
    CP_RTOL = tbot.RTOL
    CP_e3gnn_found = tbot.e3gnn_found
    CP_e3gnn_parallel_found = tbot.e3gnn_parallel_found
    CP_OUTCAR = tbot.OUTCAR

    # test training continue
    if not os.path.exists('continue_test'):
        _ = sevenn_continue_test()
        bar()
    else:
        print('Continue training test skipped')
        bar()
    # test inference same
    if not os.path.exists('inference') or not os.path.exists(
        'inference_small'
    ):
        ref_atoms = sevenn_inference_test_with_compare()
        bar()
    else:
        ref_atoms = tbot.sevenn_infer_to_atoms('inference_small')[0]
        print('Inference test skipped')
        bar()

    # test get_model serial
    if not os.path.exists('potential/serial.pt'):
        seri_pot_str = tbot.sevenn_get_model_test(
            CHECKPOINT, is_parallel=False
        )
        bar()
    else:
        seri_pot_str = os.path.abspath('potential/serial.pt')
        print('Potential file already exists')
        bar()

    # test lammps inference same
    if CP_e3gnn_found and not os.path.exists('e3gnn_test'):
        print('Test lammps e3gnn (serial)')
        lmp_serial_atoms = tbot.sevenn_lmp_test(
            ref_atoms, 'e3gnn', seri_pot_str, './e3gnn_test'
        )
        bar()
        if tbot.compare_atoms(ref_atoms, lmp_serial_atoms, CP_ATOL, CP_RTOL):
            print('LAMMPS e3gnn serial test passed')
        bar()
    else:
        print('LAMMPS e3gnn serial test skipped')
        bar()

    # test get_model parallel
    if not os.path.exists('potential/parallel_0.pt'):
        para_pot_str = tbot.sevenn_get_model_test(CHECKPOINT, is_parallel=True)
        bar()
    else:
        pts = sorted(glob.glob('./potential/parallel*.pt'))
        para_pot_str = f'{len(pts)}'
        for pt in pts:
            para_pot_str += f' {os.path.abspath(pt)}'
        print('Parallel potential files already exists')
        bar()

    # test parallel lammps inference same
    if CP_e3gnn_parallel_found and not os.path.exists('e3gnn_parallel_211'):
        print('Replicate cell by 2x1x1 for parallel test')
        serial_211 = tbot.sevenn_lmp_test(
            ref_atoms, 'e3gnn', seri_pot_str, './e3gnn_211', replicate='2 1 1'
        )
        bar()
        print('Test lammps e3gnn/parallel')
        para_211 = tbot.sevenn_lmp_test(
            ref_atoms,
            'e3gnn/parallel',
            para_pot_str,
            './e3gnn_parallel_211',
            mpicmd='mpirun',
            replicate='2 1 1',
            ncores=2,
        )
        bar()
        if tbot.compare_atoms(serial_211, para_211, CP_ATOL, CP_RTOL):
            print('LAMMPS e3gnn parallel test passed')
        bar()
    else:
        print('LAMMPS e3gnn parallel test skipped')
        bar()


if __name__ == '__main__':
    main()
