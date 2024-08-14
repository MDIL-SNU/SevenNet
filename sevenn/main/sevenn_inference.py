import argparse
import os
import sys

import torch

from sevenn._const import SEVENN_VERSION
from sevenn.scripts.inference import inference_main

description = (
    f'sevenn version={SEVENN_VERSION}, sevenn_inference. '
    + 'Evaluate sevenn_data/POSCARs/OUTCARs '
    + 'using the model stored in a checkpoint.'
)
checkpoint_help = 'checkpoint'
target_help = 'target files to evaluate. '


def main(args=None):
    checkpoint, target, device, ncores, output, batch, modal, =\
        cmd_parse_data(args)
    if not os.path.exists(checkpoint):
        print(f'{checkpoint} does not exist')
        sys.exit(1)
    if not os.path.exists(output):
        os.makedirs(output)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inference_main(checkpoint, target, output, ncores, device, batch, modal)


def cmd_parse_data(args=None):
    ag = argparse.ArgumentParser(description=description)
    ag.add_argument('checkpoint', type=str, help=checkpoint_help)
    ag.add_argument('target', type=str, nargs='+', help=target_help)
    ag.add_argument(
        '-d',
        '--device',
        type=str,
        default='auto',
        help='cpu/cuda/cuda:x',
    )
    ag.add_argument(
        '-n',
        '--ncores',
        type=int,
        default=1,
        help='if cpu, number of cores to use',
    )
    ag.add_argument(
        '-o',
        '--output',
        type=str,
        default='sevenn_inference_result',
        help='path to save the outputs',
    )
    ag.add_argument('-b', '--batch', type=int, default='5', help='batch size')

    ag.add_argument(
        '-m',
        '--modal',
        type=str,
        default='common',
        help='modality for multi-modal inference',
    )

    args = ag.parse_args()

    checkpoint = args.checkpoint
    target = args.target

    device = args.device
    ncores = args.ncores
    output = args.output
    batch = args.batch
    modal = args.modal
    return checkpoint, target, device, ncores, output, batch, modal
