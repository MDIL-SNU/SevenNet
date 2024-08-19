import argparse
import os

import sevenn.util
from sevenn import __version__
from sevenn.scripts.deploy import deploy, deploy_parallel

description_get_model = (
    f'sevenn version={__version__}, sevenn_get_model.'
    + ' Deploy model for LAMMPS from the checkpoint'
)
checkpoint_help = (
    'path to the checkpoint | SevenNet-0 | 7net-0 |'
    ' {SevenNet-0|7net-0}_{11July2024|22May2024}'
)
output_name_help = 'filename prefix'
get_parallel_help = 'deploy parallel model'


def main(args=None):
    checkpoint, output_prefix, get_parallel = cmd_parse_get_model(args)
    get_serial = not get_parallel

    if output_prefix is None:
        output_prefix = (
            'deployed_parallel' if not get_serial else 'deployed_serial'
        )

    checkpoint_path = None
    if os.path.isfile(checkpoint):
        checkpoint_path = checkpoint
    else:
        checkpoint_path = sevenn.util.pretrained_name_to_path(checkpoint)

    model, config = sevenn.util.model_from_checkpoint(checkpoint_path)
    stct_dct = model.state_dict()

    if get_serial:
        deploy(stct_dct, config, output_prefix)
    else:
        deploy_parallel(stct_dct, config, output_prefix)


def cmd_parse_get_model(args=None):
    ag = argparse.ArgumentParser(description=description_get_model)
    ag.add_argument('checkpoint', help=checkpoint_help, type=str)
    ag.add_argument(
        '-o', '--output_prefix', nargs='?', help=output_name_help, type=str
    )
    ag.add_argument(
        '-p', '--get_parallel', help=get_parallel_help, action='store_true'
    )
    args = ag.parse_args()
    checkpoint = args.checkpoint
    output_prefix = args.output_prefix
    get_parallel = args.get_parallel
    return checkpoint, output_prefix, get_parallel
