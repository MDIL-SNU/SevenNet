import argparse
import os

import torch

from sevenn import __version__
from sevenn.scripts.inference import inference_main

description = (
    f'sevenn version={__version__}, sevenn_inference. '
    + 'Evaluate sevenn_data/POSCARs/OUTCARs/ase readable '
    + 'using the model stored in a checkpoint.'
)
checkpoint_help = 'checkpoint'
target_help = 'target files to evaluate. '


def main(args=None):
    args = cmd_parse_data(args)
    torch.set_num_threads(args.nthreads)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inference_main(
        args.checkpoint,
        args.target,
        args.output,
        args.nthreads,
        args.nworkers,
        device,
        args.batch,
    )


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
        '--nthreads',
        type=int,
        default=1,
        help='passed to torch.set_num_threads',
    )
    ag.add_argument(
        '-nw',
        '--nworkers',
        type=int,
        default=1,
        help='passed to dataloader, if given with -ofg',
    )
    ag.add_argument(
        '-o',
        '--output',
        type=str,
        default='sevenn_inference_result',
        help='path to save the outputs',
    )
    ag.add_argument(
        '-b',
        '--batch',
        type=int,
        default='5',
        help='batch size'
    )
    ag.add_argument(
        '-ofg',
        '--on_the_fly_graph_build',
        action='store_true',
        help='build graph on the fly to save memory'
    )

    args = ag.parse_args()

    return args
