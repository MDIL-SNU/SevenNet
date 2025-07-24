import argparse
import os

from sevenn import __version__

description = (
    'print the selected preset for training. '
    + 'ex) sevennet_preset fine_tune > my_input.yaml'
)

preset_help = 'Name of preset'


def add_parser(subparsers):
    ag = subparsers.add_parser('preset', help=description)
    add_args(ag)


def add_args(parser):
    ag = parser
    ag.add_argument(
        'preset', choices=[
            'fine_tune',
            'fine_tune_le',
            'sevennet-0',
            'sevennet-l3i5',
            'base',
            'multi_modal',
            'mf_ompa_fine_tune',
        ],
        help=preset_help
    )


def run(args):
    preset = args.preset
    prefix = os.path.abspath(f'{os.path.dirname(__file__)}/../presets')
    with open(f'{prefix}/{preset}.yaml', 'r') as f:
        print(f.read())


# When executed as sevenn_preset (legacy way)
def main(args=None):
    ag = argparse.ArgumentParser(description=description)
    add_args(ag)
    run(ag.parse_args())
