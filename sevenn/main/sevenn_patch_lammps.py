import argparse
import os
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
    ag.add_argument('--flashTP', help='Enable flashTP', action='store_true')
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

    if args.flashTP:
        import sevenn.nn.flash_helper

        if not sevenn.nn.flash_helper.is_flash_available():
            raise ImportError('FlashTP not installed or no GPU found.')

        import flashTP_e3nn.flashTP as hook

        flash_dir = os.path.join(
            os.path.dirname(hook.__file__), 'sptp_exp_opt_large'
        )
        print('  - FlashTP support enabled.')
    else:
        flash_dir = None

    script = f'{pair_e3gnn_dir}/patch_lammps.sh'
    cmd = f'{script} {lammps_dir} {cxx_standard} {d3_support}'

    if flash_dir is not None:
        cmd += f' {flash_dir}'

    res = subprocess.run(cmd.split())
    return res.returncode  # is it meaningless?


def main(args=None):
    ag = argparse.ArgumentParser(description=description)
    add_args(ag)
    run(ag.parse_args())


if __name__ == '__main__':
    main()
