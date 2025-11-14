import argparse
import os
import pathlib

import torch

from sevenn.logger import Logger
from sevenn.util import load_checkpoint, pretrained_name_to_path

from .lmp_mliap_wrapper import SevenNetLAMMPSMLIAPWrapper

logger = Logger(screen=True)

def main(args=None):
    # === parse inputs ===
    parser = argparse.ArgumentParser(
        description='Create SevenNet LAMMPS ML-IAP file from saved models.',
    )

    # positional arguments:
    parser.add_argument(
        'model_path',
        help=f'path to a checkpoint model or name of model to use',
        type=str,
    )

    parser.add_argument(
        'output_path',
        help='path to write SevenNet LAMMPS ML-IAP interface file (ending with .pt)',
        type=pathlib.Path,
    )

    parser.add_argument(
        '-m',
        '--modal',
        help='Channel of multi-task model',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--enable_cueq',
        help='use cueq.',
        action='store_true',
    )

    parser.add_argument(
        '--enable_flash',
        help='use flashTP.',
        action='store_true',
    )

    parser.add_argument(
        '--cutoff',
        help='Neighbor cutoff (Angstrom). Required if it cannot be inferred from the model.',
        type=float,
        default=None,
    )

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to create LAMMPS ML-IAP artefact.')

    # parse pathinfo
    args = parser.parse_args(args=args)

    model_path = args.model_path
    out_path = args.output_path
    if not str(out_path).endswith('.pt'):
        out_path = out_path.with_suffix(out_path.suffix + '.pt')
    modal = args.modal
    cutoff = args.cutoff
    use_cueq  = args.enable_cueq
    use_flash = args.enable_flash

    # === create and save ML-IAP module ===
    logger.writeline(f'Creating LAMMPS ML-IAP artefact from {model_path}...')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mliap_module = SevenNetLAMMPSMLIAPWrapper(
        model_path=model_path,
        modal=modal,
        enable_cueq=use_cueq,
        enable_flash=use_flash,
        cutoff=cutoff,
    )
    torch.save(mliap_module, out_path)
    logger.writeline(f'LAMMPS ML-IAP artefact saved to {out_path}')

if __name__ == '__main__':
    main()
