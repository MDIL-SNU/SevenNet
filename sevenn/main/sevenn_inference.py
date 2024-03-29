import argparse
import os
import sys

import torch

import sevenn._keys as KEY
from sevenn._const import SEVENN_VERSION
from sevenn.scripts.inference import inference_main

description = (
    f'sevenn version={SEVENN_VERSION}, sevenn_inference\n'
    + 'inference sevenn_data or POSCARs or OUTCARs from checkpoint.\n'
)
checkpoint_help = 'path to checkpoint\n'
target_help = (
    'path to sevenn_data or OUTCAR or POSCAR.\n'
    + 'Infer its data type by its extenson or filename.\n'
    + "It knows expand '*' but expect only one kind of data type\n"
)


def main(args=None):
    checkpoint, target, device, ncores, output, batch = cmd_parse_data(args)
    if not os.path.exists(checkpoint):
        print(f'{checkpoint} does not exist')
        sys.exit(1)
    if not os.path.exists(output):
        os.makedirs(output)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inference_main(checkpoint, target, output, ncores, device, batch)


def cmd_parse_data(args=None):
    ag = argparse.ArgumentParser(description=description)
    ag.add_argument('checkpoint', type=str, help=checkpoint_help)
    ag.add_argument('target', type=str, nargs='+', help=target_help)
    ag.add_argument(
        '-d',
        '--device',
        type=str,
        default='auto',
        help="device to use 'cpu/cuda/cuda:x'",
    )
    ag.add_argument(
        '-n',
        '--ncores',
        type=int,
        default=1,
        help='number of cores to use (if cpu)',
    )
    ag.add_argument(
        '-o',
        '--output',
        type=str,
        default='sevenn_infer_result',
        help='path to save results',
    )
    ag.add_argument(
        '-b', '--batch', type=int, default='5', help='batch size for inference'
    )

    args = ag.parse_args()

    checkpoint = args.checkpoint
    target = args.target

    device = args.device
    ncores = args.ncores
    output = args.output
    batch = args.batch
    return checkpoint, target, device, ncores, output, batch
