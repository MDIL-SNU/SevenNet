import argparse
import os
import os.path as osp
import subprocess

from sevenn import __version__

# python wrapper of patch_lammps.sh script
# importlib.resources is correct way to do these things
# but it changes so frequently to use
pair_e3gnn_dir = os.path.abspath(f'{os.path.dirname(__file__)}/../pair_e3gnn')

description = 'patch LAMMPS with e3gnn(7net) pair-styles before compile'


def add_parser(subparsers):
    ag = subparsers.add_parser('patch_lammps', help=description)
    add_args(ag)


def add_args(parser):
    ag = parser
    ag.add_argument('lammps_dir', help='Path to LAMMPS source', type=str)
    ag.add_argument('--d3', help='Enable D3 support', action='store_true')
    ag.add_argument(
        '--flashTP',
        '--enable_flash',
        dest='enable_flash',
        help='Enable flashTP',
        action='store_true',
    )
    ag.add_argument(
        '--oeq',
        '--enable_oeq',
        dest='enable_oeq',
        help='Enable OpenEquivariance',
        action='store_true',
    )
    ag.add_argument(
        '--atomic_stress',
        help='Patch pair_e3gnn with atomic-stress enabled source files.',
        action='store_true',
    )
    # cxx_standard is detected automatically


def run(args):
    lammps_dir = os.path.abspath(args.lammps_dir)

    print('Patching LAMMPS with the following settings:')
    print('  - LAMMPS source directory:', lammps_dir)

    cxx_standard = '17'  # always 17

    if args.d3:
        d3_support = '1'
        print('  - D3 support enabled')
    else:
        d3_support = '0'
        print('  - D3 support disabled')

    atomic_stress = '1' if args.atomic_stress else '0'
    if args.atomic_stress:
        print('  - Atomic stress patch enabled')
    else:
        print('  - Atomic stress patch disabled')

    so_oeq = ''
    if args.enable_oeq:
        try:
            from openequivariance._torch.extlib import torch_ext_so_path
        except ImportError:
            raise ImportError('OpenEquivariance import failed.')

        so_oeq = torch_ext_so_path()
        if not osp.isfile(so_oeq):
            raise ValueError(f'OEQ .so file not found: {so_oeq}')
        print(f'  - OEQ support enabled: {so_oeq}')

    so_lammps = ''
    if args.enable_flash:
        try:
            import flashTP_e3nn.flashTP as hook
        except ImportError:
            raise ImportError('FlashTP import failed.')

        flash_dir = osp.abspath(osp.dirname(hook.__file__))

        so_files = []
        so_lammps = []
        for ls in os.listdir(flash_dir):
            fpath = osp.join(flash_dir, ls)
            if ls.endswith('.so'):
                so_files.append(fpath)
                if 'lammps' in ls:
                    so_lammps.append(fpath)
        if len(so_files) == 0:
            raise ValueError(
                f'FlashTP .so file not found. The dir searched: {flash_dir}'
            )
        if len(so_lammps) == 0:
            raise ValueError(
                f'FlashTP lammps .so file not found  The dir searched: {flash_dir}'
            )
        elif len(so_lammps) > 1:
            raise ValueError(f'More than 1 lammps .so files are found: {so_lammps}')
        so_lammps = so_lammps[0]

        print('  - FlashTP support enabled.')
    else:
        flash_dir = None

    script = f'{pair_e3gnn_dir}/patch_lammps.sh'
    cmd = f'{script} {lammps_dir} {cxx_standard} {d3_support}'

    if args.enable_flash:
        assert osp.isfile(so_lammps)
        cmd += f' {so_lammps}'
    else:
        cmd += ' NONE'

    if args.enable_oeq:
        assert osp.isfile(so_oeq)
        cmd += f' {so_oeq}'
    else:
        cmd += ' NONE'

    cmd += f' {atomic_stress}'

    res = subprocess.run(cmd.split())
    return res.returncode  # is it meaningless?


def main(args=None):
    ag = argparse.ArgumentParser(description=description)
    add_args(ag)
    run(ag.parse_args())


if __name__ == '__main__':
    main()
