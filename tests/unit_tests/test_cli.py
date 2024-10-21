import csv
import os
import pathlib
from unittest import mock

import ase.io
import numpy as np
import pytest
import yaml
from ase.build import bulk

from sevenn.main.sevenn import main as sevenn_main
from sevenn.main.sevenn_get_model import main as get_model_main
from sevenn.main.sevenn_graph_build import main as graph_build_main
from sevenn.main.sevenn_inference import main as inference_main
from sevenn.sevenn_logger import Logger
from sevenn.sevennet_calculator import SevenNetCalculator
from sevenn.util import pretrained_name_to_path

main = os.path.abspath(f'{os.path.dirname(__file__)}/../../sevenn/main/')
preset = os.path.abspath(f'{os.path.dirname(__file__)}/../../sevenn/presets/')
file_path = pathlib.Path(__file__).parent.resolve()

data_root = (pathlib.Path(__file__).parent.parent / 'data').resolve()
hfo2_path = str(data_root / 'systems' / 'hfo2.extxyz')
hfo2_7net_0_inference_path = data_root / 'inferences' / 'snet0_on_hfo2'
cp_0_path = str(data_root / 'checkpoints' / 'cp_0.pth')

Logger()  # init


@pytest.fixture
def atoms_hfo():
    atoms1 = bulk('HfO', 'rocksalt', a=5.63)
    atoms1.set_cell([[1.0, 2.815, 2.815], [2.815, 0.0, 2.815], [2.815, 2.815, 0.0]])
    atoms1.set_positions([[0.0, 0.0, 0.0], [2.815, 0.0, 0.0]])
    return atoms1


@pytest.fixture(scope='module')
def sevennet_0_cal():
    return SevenNetCalculator('7net-0_11July2024')


def test_get_model_serial(tmp_path, capsys):
    output_file = tmp_path / 'mypot.pt'
    cp = pretrained_name_to_path('7net-0')
    cli_args = ['-o', str(output_file), cp]
    with mock.patch('sys.argv', [f'{main}/sevenn_get_model.py'] + cli_args):
        get_model_main()
        _ = capsys.readouterr()  # not used
        assert output_file.is_file(), '.pt file is not written'


def test_get_model_parallel(tmp_path, capsys):
    output_dir = tmp_path / 'my_parallel'
    cp = pretrained_name_to_path('7net-0')
    expected_file_cnt = 5  # 5 interaction layers
    cli_args = ['-o', str(output_dir), '-p', cp]
    with mock.patch('sys.argv', [f'{main}/sevenn_get_model.py'] + cli_args):
        # with pytest.raises(SystemExit):
        get_model_main()
        _ = capsys.readouterr()  # not used
        assert output_dir.is_dir(), 'parallel model directory not exist'
        for i in range(expected_file_cnt):
            assert (output_dir / f'deployed_parallel_{i}.pt').is_file()


@pytest.mark.parametrize('source', [(hfo2_path)])
def test_graph_build(source, tmp_path):
    output_dir = tmp_path / 'sevenn_data'
    output_f = output_dir / 'my_graph.pt'
    output_yml = output_dir / 'my_graph.yaml'
    cli_args = ['-o', str(tmp_path), '-f', 'my_graph.pt', source, '4.0']
    with mock.patch('sys.argv', [f'{main}/sevenn_graph_build.py'] + cli_args):
        graph_build_main()

        assert output_dir.is_dir()
        assert output_f.is_file()
        assert output_yml.is_file()


@pytest.mark.parametrize(
    'batch,device,save_graph',
    [
        (1, 'cpu', False),
        (2, 'cpu', False),
        (1, 'cpu', True),
    ],
)
def test_inference(batch, device, save_graph, tmp_path):
    checkpoint = '7net-0'
    target = hfo2_path
    ref_path = hfo2_7net_0_inference_path

    output_dir = tmp_path / 'inference_results'
    files = ['info.csv', 'per_graph.csv', 'per_atom.csv', 'errors.txt']
    cli_args = [
        '--output',
        str(output_dir),
        '--device',
        device,
        '--batch',
        str(batch),
        checkpoint,
        target,
    ]
    if save_graph:
        cli_args.append('--save_graph')
    with mock.patch('sys.argv', [f'{main}/sevenn_inference.py'] + cli_args):
        inference_main()

    assert output_dir.is_dir()
    for f in files:
        assert (output_dir / f).is_file()
    with open(output_dir / 'errors.txt', 'r', encoding='utf-8') as f:
        errors = [float(ll.split(':')[-1].strip()) for ll in f.readlines()]
    with open(ref_path / 'errors.txt', 'r', encoding='utf-8') as f:
        errors_ref = [float(ll.split(':')[-1].strip()) for ll in f.readlines()]
    assert np.allclose(np.array(errors), np.array(errors_ref))

    with open(output_dir / 'info.csv', 'r') as f:
        reader = csv.DictReader(f)
        for dct in reader:
            assert dct['file'] == hfo2_path
        assert reader.line_num == 3

    if save_graph:
        assert (output_dir / 'sevenn_data').is_dir()
        assert (output_dir / 'sevenn_data' / 'saved_graph.pt').is_file()
        assert (output_dir / 'sevenn_data' / 'saved_graph.yaml').is_file()


def test_inference_unlabeled(atoms_hfo, tmp_path):
    labeled = str(hfo2_path)
    unlabeled = str(tmp_path / 'unlabeled.xyz')
    ase.io.write(unlabeled, atoms_hfo)

    output_dir = tmp_path / 'inference_results'
    cli_args = ['--output', str(output_dir), cp_0_path, labeled, unlabeled]
    with mock.patch('sys.argv', [f'{main}/sevenn_inference.py'] + cli_args):
        inference_main()

    with open(output_dir / 'info.csv', 'r') as f:
        reader = csv.DictReader(f)
        for dct in reader:
            assert dct['file'] in [labeled, unlabeled]
        assert reader.line_num == 4


def test_inference_labeled_w_kwargs(atoms_hfo, tmp_path):
    atoms_hfo.info['my_energy'] = 1.0
    atoms_hfo.arrays['my_force'] = np.full((len(atoms_hfo), 3), 7.7)
    # this should be considered as Voigt, xx, yy, zz, yz, zx, xy
    atoms_hfo.info['my_stress'] = np.array([1, 2, 3, 4, 5, 6])

    unlabeled = str(tmp_path / 'unlabeled.xyz')
    ase.io.write(unlabeled, atoms_hfo)

    output_dir = tmp_path / 'inference_results'
    cli_args = [
        '--output',
        str(output_dir),
        cp_0_path,
        unlabeled,
        '--kwargs',
        'energy_key=my_energy',
        'force_key=my_force',
        'stress_key=my_stress',
    ]
    with mock.patch('sys.argv', [f'{main}/sevenn_inference.py'] + cli_args):
        inference_main()

    per_graph = None
    with open(output_dir / 'per_graph.csv', 'r') as f:
        reader = csv.DictReader(f)
        for dct in reader:
            per_graph = dct
        assert reader.line_num == 2
    assert per_graph is not None

    stress_coeff = -1602.1766208
    assert np.allclose(float(per_graph['stress_yy']), 2 * stress_coeff)
    assert np.allclose(float(per_graph['stress_yz']), 4 * stress_coeff)
    assert np.allclose(float(per_graph['stress_zx']), 5 * stress_coeff)
    assert np.allclose(float(per_graph['stress_xy']), 6 * stress_coeff)


@pytest.mark.parametrize(
    'preset_name,mode,data_path',
    [
        ('fine_tune', 'train_v2', hfo2_path),
        ('base', 'train_v2', hfo2_path),
        ('sevennet-0', 'train_v1', hfo2_path),
    ],
)
def test_sevenn_preset(preset_name, mode, data_path, tmp_path):
    preset_path = os.path.join(preset, preset_name + '.yaml')
    with open(preset_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['train']['epoch'] = 1
    if mode == 'train_v2':
        cfg['data']['load_trainset_path'] = data_path
        cfg['data'].pop('load_testset_path', None)
    elif mode == 'train_v1':
        cfg['data']['load_dataset_path'] = data_path
    else:
        assert False
    cfg['data']['load_validset_path'] = data_path

    input_yam = str(tmp_path / 'input.yaml')
    with open(input_yam, 'w') as f:
        yaml.dump(cfg, f)

    Logger().switch_file(str(tmp_path / 'log.sevenn'))
    cli_args = ['-w', str(tmp_path), '-m', mode, input_yam]
    with mock.patch('sys.argv', [f'{main}/sevenn.py'] + cli_args):
        sevenn_main()

    assert (tmp_path / 'lc.csv').is_file() or (tmp_path / 'log.csv').is_file()
    assert (tmp_path / 'log.sevenn').is_file()
    assert (tmp_path / 'checkpoint_best.pth').is_file()
