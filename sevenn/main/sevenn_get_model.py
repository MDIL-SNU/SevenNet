import argparse

import torch

from sevenn.scripts.deploy import deploy_parallel, deploy
import sevenn._const as _const
import sevenn._keys as KEY

description_get_model = \
    f"sevenn version={_const.SEVENN_VERSION}, sevenn_get_model."\
    + " Deploy model from checkpoint"
checkpoint_help = "checkpoint path to deploy model"
output_name_help = "filename prefix of deployed model"
get_parallel_help = "whether deploy parallel model"


def main(args=None):
    checkpoint, output_prefix, get_parallel = cmd_parse_get_model(args)
    get_serial = not get_parallel
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
            print(f"{k_d} was not found in given config")
            print(f"{v_d} inserted as defaults")
            config[k_d] = v_d

    if get_serial:
        deploy(stct_dct, config, output_prefix)
    else:
        #if config[KEY.NUM_CONVOLUTION] == 1:
        #    raise ValueError("parallel model of NUM_CONVOLUTION == 1 is meaningless")
        deploy_parallel(stct_dct, config, output_prefix)


def cmd_parse_get_model(args=None):
    ag = argparse.ArgumentParser(description=description_get_model)
    ag.add_argument('checkpoint', help=checkpoint_help, type=str)
    ag.add_argument('-o', '--output_prefix', nargs='?',
                    help=output_name_help, type=str)
    ag.add_argument('-p', '--get_parallel', help=get_parallel_help, action='store_true')
    args = ag.parse_args()
    checkpoint = args.checkpoint
    output_prefix = args.output_prefix
    get_parallel = args.get_parallel
    return checkpoint, output_prefix, get_parallel
