import argparse
import pathlib

import torch
import os

from .lmp_mliap_wrapper import SevenNetLAMMPSMLIAPWrapper
from sevenn.logger import Logger
from sevenn.util import pretrained_name_to_path

logger = Logger(screen=True)
DEFAULT_MODEL_NAME = "7net-0"

def main(args=None):
    # === parse inputs ===
    parser = argparse.ArgumentParser(
        description="Create SevenNet LAMMPS ML-IAP file from saved models.",
    )

    # positional arguments:
    parser.add_argument(
        "model_path",
        help=f"path to a checkpoint model or name of model to use",
        type=str,
    )

    parser.add_argument(
        "output_path",
        help="path to write SevenNet LAMMPS ML-IAP interface file (must end with `.7net.lmp.pt`)",
        type=pathlib.Path,
    )

    parser.add_argument(
        '-m',
        '--modal',
        help='Modality of multi-modal model',
        default="NONE",
        type=str,
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
        "--tf32",
        help="whether to use TF32 or not (default: False)",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    
    parser.add_argument(
        "--device",
        help="Target device for inference: 'cuda' or 'cpu'. Default: auto-detect.",
        type=str,
        default=None,
        choices=[None, "cuda", "cpu"]
    )
    
    parser.add_argument(
        "--type-to-Z",
        help="Optional mapping from LAMMPS atom types (1..ntypes) to atomic numbers, e.g., '1,8' for H,O.",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--element-types",
        help="Element symbols in LAMMPS type order, e.g. 'H,O' (must match pair_coeff order).",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--cutoff",
        help="Neighbor cutoff (Angstrom). Required if it cannot be inferred from the model. Note: rcutfac will be automatically set to cutoff * 0.5",
        type=float,
        default=None,
    )

    args = parser.parse_args(args=args)
    
    out_path: pathlib.Path = args.output_path
    if not str(out_path).endswith(".7net.lmp.pt"):
        raise ValueError(f"Output path must end with `.7net.lmp.pt`, got: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
        
    checkpoint_path = None
    if os.path.isfile(args.model_path):
        checkpoint_path = str(pathlib.Path(args.model_path).resolve())
    else:
        checkpoint_path = pretrained_name_to_path(args.model_path)
        
    element_types = args.element_types.split(",") if args.element_types else None
    type_to_Z = [int(x) for x in args.type_to_Z.split(",")] if args.type_to_Z else None

    # Auto device if not set
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    # === create and save ML-IAP module ===
    logger.writeline(f"Creating LAMMPS ML-IAP artefact from {checkpoint_path} on device={device} ...")
    
    modal = None if args.modal == "NONE" else args.modal 
    
    mliap_module = SevenNetLAMMPSMLIAPWrapper(
        model_path=str(checkpoint_path),
        tf32=bool(args.tf32),
        device=device,
        element_types=element_types,
        type_to_Z=type_to_Z,
        cutoff=args.cutoff,
        calculator_kwargs={"modal": modal, "enable_cueq": args.enable_cueq, "enable_flash": args.enable_flash},
    )
    
    torch.save(mliap_module, args.output_path)
    
    logger.writeline(f"LAMMPS ML-IAP artefact saved to {args.output_path}")
    if args.cutoff:
        logger.writeline(f"Cutoff: {args.cutoff} Å, rcutfac: {args.cutoff * 0.5} (automatically set)")


if __name__ == "__main__":
    main()
