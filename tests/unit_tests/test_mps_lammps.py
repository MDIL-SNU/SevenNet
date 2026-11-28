"""LAMMPS on the Apple GPU: the serial pair style (sevenn/mps/lammps/src/, pair.py)
and the `fix external` bridge (sevenn/mps/lammps/external.py). Tests that need
LAMMPS are skipped without the lammps Python module; MPS tests are skipped without
an Apple GPU."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk

from sevenn.calculator import SevenNetCalculator
from sevenn.mps.modules import mps_torch_supported

needs_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and mps_torch_supported()),
    reason='needs an Apple GPU and PyTorch 2.6 or later',
)


@needs_mps
def test_lammps_bridge_equals_the_calculator(tmp_path):
    lammps = pytest.importorskip('lammps')
    from sevenn.mps.lammps.external import run_input

    calc = SevenNetCalculator('7net-0', device='mps')
    a = 5.63
    at = bulk('NaCl', 'rocksalt', a=a, cubic=True)
    at.rattle(0.05, seed=2)
    data = tmp_path / 'at.data'
    from ase.io import write

    write(
        data, at, format='lammps-data', specorder=['Na', 'Cl'], atom_style='atomic'
    )
    inp = tmp_path / 'in.lmp'
    inp.write_text(f"""units metal
atom_style atomic
boundary p p p
read_data {data}
mass 1 22.99
mass 2 35.45
pair_style zero 4.0
pair_coeff * *
fix sevenn all external pf/callback 1 1
fix_modify sevenn energy yes virial yes
thermo_style custom step pe press pxx pyy pzz pxy pxz pyz
run 0
""")
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    run_input(lmp, str(inp), calc, ['Na', 'Cl'])
    pe, press = lmp.get_thermo('pe'), lmp.get_thermo('pxx')
    fl = lmp.numpy.extract_atom('f')[: len(at)].copy()
    ids = lmp.numpy.extract_atom('id')[: len(at)].copy()
    lmp.close()
    at.calc = calc
    assert abs(pe - at.get_potential_energy()) < 1e-3
    f = at.get_forces()
    assert np.abs(fl[np.argsort(ids)] - f).max() < 2e-4
    # ASE stress in eV/A^3; LAMMPS pressure in bar, kinetic part is 0 at rest
    assert abs(press - (-at.get_stress()[0] * 1602176.634)) < 2.0


_PLUGIN = str(
    Path(__file__).parents[2]
    / 'sevenn'
    / 'mps'
    / 'lammps'
    / 'src'
    / 'e3gnnmpsplugin.so'
)


needs_plugin = pytest.mark.skipif(
    not Path(_PLUGIN).exists(),
    reason='sevenn/mps/lammps/src/build_plugin.sh not run',
)


def _pair_lammps(tmp_path, at, coeff):
    lammps = pytest.importorskip('lammps')
    from ase.io import write

    data = tmp_path / 'at.data'
    write(
        data, at, format='lammps-data', specorder=['Na', 'Cl'], atom_style='atomic'
    )
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    for line in (
        f'plugin load {_PLUGIN}',
        'units metal',
        'atom_style atomic',
        'boundary p p p',
        f'read_data {data}',
        'mass 1 22.99',
        'mass 2 35.45',
        'pair_style e3gnn',
        f'pair_coeff * * {coeff}',
    ):
        lmp.command(line)
    return lmp


@needs_mps
@needs_plugin
def test_pair_style_equals_the_calculator(tmp_path):
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=2)
    lmp = _pair_lammps(tmp_path, at, '7net-0 Na Cl')
    lmp.command('compute pea all pe/atom')
    lmp.command('compute pesum all reduce sum c_pea')  # tallies per-atom energies
    lmp.command(
        'thermo_style custom step pe c_pesum press pxx pyy pzz pxy pxz pyz'
    )
    lmp.command('run 0')
    pe, press = lmp.get_thermo('pe'), lmp.get_thermo('pxx')
    ids = lmp.numpy.extract_atom('id')[: len(at)].copy()
    fl = lmp.numpy.extract_atom('f')[: len(at)].copy()
    eatom = lmp.numpy.extract_compute('pea', 1, 1)[: len(at)].copy()
    lmp.close()
    calc = SevenNetCalculator('7net-0', device='mps')
    at.calc = calc
    assert abs(pe - at.get_potential_energy()) < 1e-3
    assert np.abs(fl[np.argsort(ids)] - at.get_forces()).max() < 2e-4
    assert abs(press - (-at.get_stress()[0] * 1602176.634)) < 2.0
    assert np.abs(eatom[np.argsort(ids)] - calc.results['energies']).max() < 1e-4


@needs_mps
@needs_plugin
@pytest.mark.parametrize(
    'coeff, message',
    [
        ('7net-0 Na', 'not enough elements'),
        ('7net-0 Na Xx', 'neither a keyword'),
        ('deployed.pt Na Cl', 'deployed'),
    ],
)
def test_pair_style_clear_errors(tmp_path, coeff, message):
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    with pytest.raises(Exception, match=message):
        _pair_lammps(tmp_path, at, coeff)


class _FakeLammps:
    def __init__(self, ranks, units):
        self.ranks, self.units = ranks, units

    def extract_setting(self, name):
        return self.ranks

    def extract_global(self, name):
        return self.units

    def set_fix_external_callback(self, *args):
        pass


def test_external_bridge_refuses_ranks_and_units():
    from sevenn.mps.lammps.external import attach

    with pytest.raises(ValueError, match='one MPI rank'):
        attach(_FakeLammps(2, 'metal'), None, ['Na'])
    with pytest.raises(ValueError, match='units metal'):
        attach(_FakeLammps(1, 'real'), None, ['Na'])
    attach(_FakeLammps(1, 'metal'), None, ['Na'])


@needs_mps
def test_external_bridge_virial_on_a_slab(tmp_path):
    lammps = pytest.importorskip('lammps')
    from ase.io import write

    from sevenn.mps.lammps.external import run_input

    calc = SevenNetCalculator('7net-0', device='mps')
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.cell[2, 2] += 10.0
    at.pbc = [True, True, False]
    at.rattle(0.05, seed=2)
    at.positions += 0.3
    write(
        tmp_path / 'at.data',
        at,
        format='lammps-data',
        specorder=['Na', 'Cl'],
        atom_style='atomic',
    )
    inp = tmp_path / 'in.lmp'
    inp.write_text(f"""units metal
atom_style atomic
boundary p p f
read_data {tmp_path / 'at.data'}
mass 1 22.99
mass 2 35.45
pair_style zero 5.0
pair_coeff * *
fix sevenn all external pf/callback 1 1
fix_modify sevenn energy yes virial yes
thermo_style custom step pe pxx
run 0
""")
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    run_input(lmp, str(inp), calc, ['Na', 'Cl'])
    press = lmp.get_thermo('pxx')
    x = lmp.numpy.extract_atom('x')[: len(at)].copy()
    ids = lmp.numpy.extract_atom('id')[: len(at)].copy()
    lmp.close()
    assert abs(press) > 1.0
    at.positions = x[np.argsort(ids)]
    at.calc = calc
    assert abs(press - (-at.get_stress()[0] * 1602176.634)) < 2.0


@needs_mps
@needs_plugin
@pytest.mark.parametrize(
    'line, message',
    [('units real', 'units metal'), ('neigh_modify exclude type 1 2', 'exclude')],
)
def test_pair_style_refuses_what_it_cannot_honour(tmp_path, line, message):
    lammps = pytest.importorskip('lammps')
    from ase.io import write

    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    write(
        tmp_path / 'at.data',
        at,
        format='lammps-data',
        specorder=['Na', 'Cl'],
        atom_style='atomic',
    )
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    lines = [
        f'plugin load {_PLUGIN}',
        'units metal',
        'atom_style atomic',
        f'read_data {tmp_path / "at.data"}',
        'mass 1 22.99',
        'mass 2 35.45',
        'pair_style e3gnn',
        'pair_coeff * * 7net-0 Na Cl',
    ]
    if line == 'units real':
        lines[1] = line
    else:
        lines.append(line)
    for l in lines:
        lmp.command(l)
    with pytest.raises(Exception, match=message):
        lmp.command('run 0')


@needs_mps
@needs_plugin
def test_pair_style_refuses_an_empty_box():
    lammps = pytest.importorskip('lammps')
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    for l in (
        f'plugin load {_PLUGIN}',
        'units metal',
        'atom_style atomic',
        'region b block 0 10 0 10 0 10',
        'create_box 2 b',
        'mass 1 22.99',
        'mass 2 35.45',
        'pair_style e3gnn',
        'pair_coeff * * 7net-0 Na Cl',
    ):
        lmp.command(l)
    with pytest.raises(Exception, match='at least one atom'):
        lmp.command('run 0')


def test_pair_coeff_parser_says_what_is_wrong():
    from sevenn.mps.lammps.pair import _parse

    with pytest.raises(ValueError, match='modal requires'):
        _parse(['modal'])
    # an unknown keyword is reported, not taken as an element
    with pytest.raises(ValueError, match='neither a keyword'):
        _parse(['field', '1', '0', '0', 'Na', 'Cl'])
    assert _parse(['modal', 'mpa', 'Zr', 'O'])['symbols'] == ['Zr', 'O']


@needs_mps
def test_driver_with_no_atoms_gives_zero():
    from sevenn.mps.lammps.pair import PairDriver

    box = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), 0.0, 0.0, 0.0, (0, 0, 0))
    driver = PairDriver(['7net-0', 'Na', 'Cl'], 2, box)
    assert (
        driver.compute(
            0,
            memoryview(b''),
            memoryview(bytearray()),
            memoryview(b''),
            None,
            memoryview(bytearray(48)),
            box,
        )
        == 0.0
    )


def test_run_input_joins_continued_lines(tmp_path):
    from sevenn.mps.lammps.external import run_input

    class Fake(_FakeLammps):
        def __init__(self):
            super().__init__(1, 'metal')
            self.seen = []

        def command(self, line):
            self.seen.append(line)

    path = tmp_path / 'in.lmp'
    path.write_text(
        'units metal  # comment\n\nfix sevenn all external &\n   pf/callback 1 '
        '1\nprint "Step #1"\n'
    )
    lmp = Fake()
    run_input(lmp, str(path), None, ['Na'])
    assert any(
        ' '.join(c.split()) == 'fix sevenn all external pf/callback 1 1'
        for c in lmp.seen
    )
    assert 'print "Step #1"' in lmp.seen


def _apple_route() -> bool:
    try:
        from sevenn.mps.lammps.patch import detected

        return detected(False)
    except Exception:
        return False


@pytest.mark.skipif(not _apple_route(), reason='the Apple GPU route of patch_lammps')
def test_patch_lammps_reports_failure():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2]))
    env.pop('SEVENN_DISABLE_MPS', None)
    p = subprocess.run(
        [sys.executable, '-m', 'sevenn.main.sevenn_patch_lammps', '/no/such/folder'],
        capture_output=True,
        text=True,
        env=env,
    )
    assert p.returncode != 0
    assert 'Apple GPU pair style' in p.stdout


def test_patch_lammps_on_a_mac_picks_the_apple_gpu_unless_disabled(monkeypatch):
    from sevenn.mps import _policy
    from sevenn.mps.lammps.patch import detected

    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(_policy, '_TORCH_OK', True)
    monkeypatch.delenv('SEVENN_DISABLE_MPS', raising=False)
    assert detected(False)
    assert not detected(True)  # --d3, --enable_flash or --enable_oeq
    monkeypatch.setattr(_policy, '_TORCH_OK', False)  # a PyTorch before 2.6
    assert not detected(False)
    monkeypatch.setattr(_policy, '_TORCH_OK', True)
    monkeypatch.setenv('SEVENN_DISABLE_MPS', '1')
    assert not detected(False)
    monkeypatch.delenv('SEVENN_DISABLE_MPS')
    monkeypatch.setattr(sys, 'platform', 'linux')
    assert not detected(False)


@pytest.mark.parametrize(
    'mode, message',
    [('raise', 'model failed'), ('bad_shape', 'forces of shape')],
)
def test_external_bridge_stops_when_the_calculator_fails(tmp_path, mode, message):
    lammps = pytest.importorskip('lammps')
    from ase.calculators.calculator import Calculator, all_changes
    from ase.io import write

    from sevenn.mps.lammps.external import run_input

    class Failing(Calculator):
        implemented_properties = ['energy', 'forces', 'stress']
        calls = 0

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            Failing.calls += 1
            forces = np.zeros((len(atoms), 3))
            if Failing.calls == 3:
                if mode == 'raise':
                    raise RuntimeError('model failed')
                forces = np.zeros((len(atoms), 2))  # fails when published
            self.results = dict(energy=0.0, forces=forces, stress=np.zeros(6))

    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    write(
        tmp_path / 'at.data',
        at,
        format='lammps-data',
        specorder=['Na', 'Cl'],
        atom_style='atomic',
    )
    inp = tmp_path / 'in.lmp'
    inp.write_text(f"""units metal
atom_style atomic
boundary p p p
read_data {tmp_path / 'at.data'}
mass 1 22.99
mass 2 35.45
pair_style zero 4.0
pair_coeff * *
velocity all create 300 1
fix nve all nve
fix sevenn all external pf/callback 1 1
fix_modify sevenn energy yes virial yes
run 20
print "not reached"
""")
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    with pytest.raises(RuntimeError, match=message):
        run_input(lmp, str(inp), Failing(), ['Na', 'Cl'])
    assert lmp.extract_global('ntimestep') < 5 and Failing.calls == 3
    lmp.close()


@needs_mps
@needs_plugin
def test_pair_style_multi_fidelity(tmp_path):
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=5)
    lmp = _pair_lammps(tmp_path, at, '7net-mf-0 modal PBE Na Cl')
    lmp.command('run 0')
    pe = lmp.get_thermo('pe')
    lmp.close()
    at.calc = SevenNetCalculator('7net-mf-0', device='mps', modal='PBE')
    assert abs(pe - at.get_potential_energy()) < 1e-3


@needs_mps
@needs_plugin
@pytest.mark.parametrize('case', ['include', 'hybrid'])
def test_pair_style_refuses_include_and_hybrid(tmp_path, case):
    lammps = pytest.importorskip('lammps')
    from ase.io import write

    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    write(
        tmp_path / 'at.data',
        at,
        format='lammps-data',
        specorder=['Na', 'Cl'],
        atom_style='atomic',
    )
    lines = [f'plugin load {_PLUGIN}', 'units metal', 'atom_style atomic']
    if case == 'include':
        lines.append('atom_modify first na')
    lines += [
        f'read_data {tmp_path / "at.data"}',
        'mass 1 22.99',
        'mass 2 35.45',
        'group na type 1',
    ]
    if case == 'include':
        lines += [
            'pair_style e3gnn',
            'pair_coeff * * 7net-0 Na Cl',
            'neigh_modify include na',
        ]
    else:
        lines += [
            'pair_style hybrid e3gnn zero 5.0',
            'pair_coeff * * e3gnn 7net-0 Na Cl',
            'pair_coeff 1 1 zero',  # both sub-styles in use: the check is ours
        ]
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    for line in lines:
        lmp.command(line)
    message = (
        'neigh_modify include'
        if case == 'include'
        else 'cannot be a sub-style of pair_style hybrid'
    )
    with pytest.raises(Exception, match=message):
        lmp.command('run 0')
    lmp.close()


@needs_mps
@needs_plugin
def test_pair_style_pressure_tensor_on_a_tilted_cell(tmp_path):
    from ase import Atoms

    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.set_cell(at.cell + [[0, 0, 0], [0.7, 0, 0], [0.4, -0.3, 0]], scale_atoms=True)
    at.rattle(0.05, seed=6)
    lmp = _pair_lammps(tmp_path, at, '7net-0 Na Cl')
    lmp.command('thermo_style custom step pe pxx pyy pzz pxy pxz pyz')
    calc = SevenNetCalculator('7net-0', device='mps')

    def in_lammps_frame():
        # the structure as the pair style sees it: LAMMPS's frame, after any motion
        lo, hi, xy, yz, xz, _, _ = lmp.extract_box()
        ids = lmp.numpy.extract_atom('id')[: len(at)].copy()
        order = np.argsort(ids)
        x = lmp.numpy.extract_atom('x')[: len(at)].copy()[order]
        a, b, c = (hi[k] - lo[k] for k in range(3))
        cell = [[a, 0, 0], [xy, b, 0], [xz, yz, c]]
        ref = Atoms(at.get_chemical_symbols(), positions=x, cell=cell, pbc=True)
        ref.calc = calc
        return ref, lmp.numpy.extract_atom('f')[: len(at)].copy()[order]

    lmp.command('run 0')
    press = [lmp.get_thermo(k) for k in ('pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz')]
    ref, _ = in_lammps_frame()
    s = ref.get_stress(voigt=True) * 1602176.634  # xx yy zz yz xz xy, bar
    want = [-s[0], -s[1], -s[2], -s[5], -s[4], -s[3]]
    assert abs(want[3]) > 10.0  # a real shear component
    assert np.abs(np.array(press) - np.array(want)).max() < 2.0
    # a few MD steps: atoms move, and may leave the box between re-neighbouring
    for line in ('velocity all create 600 3', 'fix md all nve', 'run 5'):
        lmp.command(line)
    pe = lmp.get_thermo('pe')
    ref, fl = in_lammps_frame()
    lmp.close()
    assert abs(pe - ref.get_potential_energy()) < 1e-3
    assert np.abs(fl - ref.get_forces()).max() < 2e-4
