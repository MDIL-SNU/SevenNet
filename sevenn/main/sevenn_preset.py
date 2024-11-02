import argparse
import os

from sevenn import __version__

description_preset = (
    f'sevenn version={__version__}, sevenn_preset.'
    + ' copy paste preset training yaml file to current directory'
    + ' ex) sevennet_preset fine_tune > my_input.yaml'
)

preset_help = 'Name of preset'


def main(args=None):
    preset = cmd_parse_preset(args)
    prefix = os.path.abspath(f'{os.path.dirname(__file__)}/../presets')

    with open(f'{prefix}/{preset}.yaml', 'r') as f:
        print(f.read())


def cmd_parse_preset(args=None):
    ag = argparse.ArgumentParser(description=description_preset)
    ag.add_argument(
        'preset', choices=[
            'fine_tune',
            'sevennet-0',
            'base',
            'fine_tune_v1',
            'base_v1'
        ],
        help=preset_help
    )
    args = ag.parse_args()
    return args.preset
