import argparse

import torch

from sevenn._const import SEVENN_VERSION
from sevenn.scripts.deploy import deploy_parallel
import sevenn._keys as KEY

description_get_parallel = f"sevenn version={SEVENN_VERSION}, get_parallel"
checkpoint_help = "checkpoint to make parallel model"
output_name_help = "prefix of parallel model file"


def main(args=None):
    checkpoint, output_prefix = cmd_parse_get_parallel(args)
    cp_file = torch.load(checkpoint, map_location=torch.device('cpu'))
    if output_prefix is None:
        output_prefix = "deployed_parallel"

    config = cp_file['config']
    stct_dct = cp_file['model_state_dict']
    if config[KEY.NUM_CONVOLUTION] == 1:
        raise ValueError("parallel model of NUM_CONVOLUTION == 1 is meaningless")

    """
    import sevenn._const as _const
    import copy
    from sevenn.scripts.deploy import deploy
    defaults = {}
    defaults.update(_const.DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG)
    defaults.update(_const.DEFAULT_DATA_CONFIG)
    defaults.update(_const.DEFAULT_TRAINING_CONFIG)
    for k_d, v_d in defaults.items():
        if k_d not in config.keys():
            print(f"{k_d} was not found in givne config")
            print(f"{v_d} inserted as defaults")
            config[k_d] = v_d
    stct_cp = copy.deepcopy(stct_dct)
    deploy(stct_cp, config, "deployed_serial.pt")
    """

    deploy_parallel(stct_dct, config, output_prefix)


def cmd_parse_get_parallel(args=None):
    ag = argparse.ArgumentParser(description=description_get_parallel)
    ag.add_argument('checkpoint', help=checkpoint_help, type=str)
    ag.add_argument('-o', '--output_prefix', nargs='?',
                    help=output_name_help, type=str)
    args = ag.parse_args()
    checkpoint = args.checkpoint
    output_prefix = args.output_prefix
    return checkpoint, output_prefix
