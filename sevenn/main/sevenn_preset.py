import os
import argparse

import sevenn._const as _const

description_preset = (
    f'sevenn version={_const.SEVENN_VERSION}, sevenn_preset.'
    + ' copy paste preset training yaml file to current directory'
    + ' ex) sevennet_preset fine_tune > my_input.yaml'
)

preset_help = "Name of preset"


def main(args=None):
    preset = cmd_parse_preset(args)
    prefix = os.path.abspath(f'{os.path.dirname(__file__)}/../presets')

    with open(f"{prefix}/{preset}.yaml", "r") as f:
        print(f.read())


def cmd_parse_preset(args=None):
    ag = argparse.ArgumentParser(description=description_preset)
    ag.add_argument(
        'preset', choices=['fine_tune', 'sevennet-0', 'base'],
        help = preset_help
    )
    args = ag.parse_args()
    return args.preset
