import pytest


def pytest_addoption(parser):
    parser.addoption('--lammps_cmd', default=None, help='Lammps binary to test')
    parser.addoption(
        '--mpirun_cmd', default=None, help='mpirun binary to test parallel'
    )


@pytest.fixture
def lammps_cmd(request):
    bin = request.config.getoption('lammps_cmd')
    if bin is None:
        pytest.skip('No LAMMPS binary given, skipping test')
    return bin


@pytest.fixture
def mpirun_cmd(request):
    bin = request.config.getoption('mpirun_cmd')
    if bin is None:
        pytest.skip('No mpirun cmd given, skipping test')
    return bin
