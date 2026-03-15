import argparse
import glob
import os
import sys

description = (
    'evaluate sevenn_data/ase readable with a model (checkpoint).'
)
checkpoint_help = 'Checkpoint or pre-trained model name'
target_help = 'Target files to evaluate'


def add_parser(subparsers):
    ag = subparsers.add_parser('inference', help=description, aliases=['inf'])
    add_args(ag)


def add_args(parser):
    ag = parser
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
        '-au',
        '--allow_unlabeled',
        action='store_true',
        help='Allow energy or force unlabeled data'
    )
    ag.add_argument(
        '-m',
        '--modal',
        type=str,
        default=None,
        help='modality for multi-modal inference',
    )
    ag.add_argument(
        '-cueq',
        '--enable_cueq',
        help='use cuEquivariance to accelerate inference',
        action='store_true',
    )
    ag.add_argument(
        '-flashTP',
        '--enable_flash',
        dest='enable_flash',
        help='use FlashTP to accelerate inference',
        action='store_true',
    )
    ag.add_argument(
        '-oeq',
        '--enable_oeq',
        help='use OpenEquivariance to accelerate inference',
        action='store_true',
    )
    ag.add_argument(
        '--kwargs',
        nargs=argparse.REMAINDER,
        help='will be passed to reader, or can be used to specify EFS key',
    )


def run(args):
    import torch

    from sevenn.scripts.inference import inference
    from sevenn.util import pretrained_name_to_path

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

    if args.save_graph and args.allow_unlabeled:
        raise ValueError('save_graph and allow_unlabeled are mutually exclusive')

    if args.enable_cueq:
        from sevenn.nn.cue_helper import is_cue_available
        if not is_cue_available():
            raise ImportError('cuEquivariance not installed or no GPU found.')

    if args.enable_flash:
        from sevenn.nn.flash_helper import is_flash_available
        if not is_flash_available():
            raise ImportError('FlashTP not installed or no GPU found.')

    if args.enable_oeq:
        from sevenn.nn.oeq_helper import is_oeq_available
        if not is_oeq_available():
            raise ImportError('OpenEquivariance not installed or no GPU found.')

    inference(
        cp,
        targets,
        out,
        args.nworkers,
        device,
        args.batch,
        args.save_graph,
        args.allow_unlabeled,
        args.modal,
        enable_cueq=args.enable_cueq,
        enable_flash=args.enable_flash,
        enable_oeq=args.enable_oeq,
        **fmt_kwargs,
    )


def main(args=None):
    ag = argparse.ArgumentParser(description=description)
    add_args(ag)
    run(ag.parse_args())
