import argparse
import glob
import os
import sys

import torch

from sevenn import __version__
from sevenn.scripts.inference import inference
from sevenn.util import pretrained_name_to_path

description = (
    f'sevenn version={__version__}, sevenn_inference. '
    + 'Evaluate sevenn_data/ase readable '
    + 'using the model stored in a checkpoint.'
)
checkpoint_help = 'Checkpoint or pre-trained model name (7net-0)'
target_help = 'Target files to evaluate'


def main(args=None):
    args = cmd_parse_data(args)
    out = args.output

    if os.path.exists(out):
        raise FileExistsError(f'Directory {out} already exists')

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    targets = []
    for target in args.targets:
        targets.extend(glob.glob(target))

    if len(targets) == 0:
        print('No targets (data to inference) are found')
        sys.exit(0)

    cp = args.checkpoint
    if not os.path.isfile(cp):
        cp = pretrained_name_to_path(cp)  # raises value error

    fmt_kwargs = {}
    if args.kwargs:
        for kwarg in args.kwargs:
            k, v = kwarg.split('=')
            fmt_kwargs[k] = v

    inference(
        cp,
        targets,
        out,
        args.nworkers,
        device,
        args.batch,
        args.save_graph,
        **fmt_kwargs,
    )


def cmd_parse_data(args=None):
    ag = argparse.ArgumentParser(description=description)
    ag.add_argument('checkpoint', type=str, help=checkpoint_help)
    ag.add_argument('targets', type=str, nargs='+', help=target_help)
    ag.add_argument(
        '-d',
        '--device',
        type=str,
        default='auto',
        help='cpu/cuda/cuda:x',
    )
    ag.add_argument(
        '-nw',
        '--nworkers',
        type=int,
        default=1,
        help='Number of cores to build graph, defaults to 1',
    )
    ag.add_argument(
        '-o',
        '--output',
        type=str,
        default='./inference_results',
        help='A directory name to write outputs',
    )
    ag.add_argument(
        '-b',
        '--batch',
        type=int,
        default='4',
        help='batch size, useful for GPU'
    )
    ag.add_argument(
        '-s',
        '--save_graph',
        action='store_true',
        help='Additionally, save preprocessed graph as sevenn_data'
    )
    ag.add_argument(
        '--kwargs',
        nargs=argparse.REMAINDER,
        help='will be passed to reader, or can be used to specify EFS key',
    )

    args = ag.parse_args()

    return args
