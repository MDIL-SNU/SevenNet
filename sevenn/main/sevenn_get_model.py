import argparse
import os

import torch

import sevenn._const as _const
import sevenn.util
import sevenn._keys as KEY
from sevenn.scripts.convert_model_modality import get_single_modal_model_dct
from sevenn.scripts.deploy import deploy, deploy_parallel

description_get_model = (
    f'sevenn version={_const.SEVENN_VERSION}, sevenn_get_model.'
    + ' Deploy model for LAMMPS from the checkpoint'
)
checkpoint_help = (
    'path to the checkpoint | SevenNet-0 | 7net-0 |'
    ' {SevenNet-0|7net-0}_{11July2024|22May2024}'
)
output_name_help = 'filename prefix'
get_parallel_help = 'deploy parallel model'


def main(args=None):
    checkpoint, output_prefix, get_parallel, modal, save_cp = cmd_parse_get_model(args)
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

    if KEY.USE_MODALITY in config.keys() and config[KEY.USE_MODALITY]:
        stct_dct = get_single_modal_model_dct(stct_dct, config, modal)
        output_prefix = modal + '_' + output_prefix
        if save_cp:
            cp_file = torch.load(checkpoint_path, map_location='cpu')
            cp_file.update({'model_state_dict': stct_dct, 'config': config})
            torch.save(cp_file, checkpoint_path.replace('.', f'_{modal}.'))

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
    ag.add_argument(
        '-m',
        '--modal',
        help='Modality of multi-modal model',
        default='common',
        type=str,
    )
    ag.add_argument(
        '-s',
        '--save_checkpoint',
        help='Save converted checkpoint',
        action='store_true',
    )
    args = ag.parse_args()
    checkpoint = args.checkpoint
    output_prefix = args.output_prefix
    get_parallel = args.get_parallel
    modal = args.modal
    save_cp = args.save_checkpoint
    return checkpoint, output_prefix, get_parallel, modal, save_cp
