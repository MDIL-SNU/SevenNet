import argparse

import torch

from sevenn._const import SEVENN_VERSION
from sevenn.scripts.deploy import deploy_parallel, deploy
import sevenn._const as _const
import sevenn._keys as KEY

description_get_parallel = \
    f"sevenn version={SEVENN_VERSION}, sevenn_get_parallel"\
    + "deploy parallel model from checkpoint"
checkpoint_help = "checkpoint to make parallel model"
output_name_help = "prefix of parallel model file"
get_serial_help = "deploy serial model instead of parallel one"


def main(args=None):
    checkpoint, output_prefix, get_serial = cmd_parse_get_parallel(args)
    cp_file = torch.load(checkpoint, map_location=torch.device('cpu'))

    if output_prefix is None:
        output_prefix = "deployed_parallel" if not get_serial else "deployed_serial"

    config = cp_file['config']
    stct_dct = cp_file['model_state_dict']

    defaults = {}
    defaults.update(_const.DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG)
    defaults.update(_const.DEFAULT_DATA_CONFIG)
    defaults.update(_const.DEFAULT_TRAINING_CONFIG)
    for k_d, v_d in defaults.items():
        if k_d not in config.keys():
            print(f"{k_d} was not found in givne config")
            print(f"{v_d} inserted as defaults")
            config[k_d] = v_d

    if get_serial:
        deploy(stct_dct, config, output_prefix)
    else:
        if config[KEY.NUM_CONVOLUTION] == 1:
            raise ValueError("parallel model of NUM_CONVOLUTION == 1 is meaningless")
        deploy_parallel(stct_dct, config, output_prefix)


def cmd_parse_get_parallel(args=None):
    ag = argparse.ArgumentParser(description=description_get_parallel)
    ag.add_argument('checkpoint', help=checkpoint_help, type=str)
    ag.add_argument('-o', '--output_prefix', nargs='?',
                    help=output_name_help, type=str)
    ag.add_argument('-s', '--get_serial', help=get_serial_help, action='store_true')
    args = ag.parse_args()
    checkpoint = args.checkpoint
    output_prefix = args.output_prefix
    get_serial = args.get_serial
    return checkpoint, output_prefix, get_serial
