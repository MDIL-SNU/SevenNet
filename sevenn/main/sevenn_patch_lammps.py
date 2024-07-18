import os
import argparse
import subprocess

# python wrapper of patch_lammps.sh script

# importlib.resources is correct way to do these things
# but it changes so frequently to use 
pair_e3gnn_dir = os.path.abspath(f'{os.path.dirname(__file__)}/../pair_e3gnn')

from sevenn._const import SEVENN_VERSION

description = (
    f'sevenn version={SEVENN_VERSION}, patch LAMMPS for pair_e3gnn styles'
)


def main(args=None):
    lammps_dir = cmd_parse_main(args)
    script = f"{pair_e3gnn_dir}/patch_lammps.sh"
    cmd = f"{script} {lammps_dir}"
    res = subprocess.run(cmd.split())
    return res.returncode  # is it meaningless?


def cmd_parse_main(args=None):
    ag = argparse.ArgumentParser(description=description)
    ag.add_argument('lammps_dir', help="Path to LAMMPS source", type=str)
    args = ag.parse_args()
    return args.lammps_dir


if __name__ == '__main__':
    main()
