import argparse
import os

import torch

from sevenn._const import DEFAULT_RADIAL_TABLE_KNOTS, DEFAULT_RADIAL_TABLE_R_MIN

description_get_model = (
    'deploy LAMMPS model from the checkpoint'
)
checkpoint_help = (
    'Pretrained model name (7net-omni, 7net-nano-5.5, etc.) or path to '
    'checkpoint file. See documentation for all available models.'
)
output_name_help = 'filename prefix'
get_parallel_help = 'deploy parallel model'


def add_parser(subparsers):
    ag = subparsers.add_parser(
        'get_model', help=description_get_model, aliases=['deploy']
    )
    add_args(ag)


def add_args(parser):
    ag = parser
    ag.add_argument('checkpoint', help=checkpoint_help, type=str)
    ag.add_argument(
        '-o', '--output_prefix', nargs='?', help=output_name_help, type=str
    )
    ag.add_argument(
        '-p', '--get_parallel', help=get_parallel_help, action='store_true'
    )
    ag.add_argument(
        '--compact_checkpoint',
        help='Write a compact inference .pth. Use -o to name the output.',
        action='store_true',
    )
    ag.add_argument(
        '-m',
        '--modal',
        help='Modality of multi-modal model',
        type=str,
    )
    ag.add_argument(
        '-flashTP',
        '--enable_flash',
        '--enable_flashTP',
        dest='enable_flash',
        help='use flashTP. LAMMPS must be specially compiled.',
        action='store_true',
    )
    ag.add_argument(
        '-cueq',
        '--enable_cueq',
        help='use cuEquivariance. Only support ML-IAP interface.',
        action='store_true',
    )
    ag.add_argument(
        '-oeq',
        '--enable_oeq',
        help='use OpenEquivariance. LAMMPS must be specially compiled.',
        action='store_true',
    )
    ag.add_argument(
        '-lunatp',
        '--enable_lunatp',
        help='use LunaTP. Only support ML-IAP interface.',
        action='store_true',
    )
    ag.add_argument(
        '-mliap',
        '--use_mliap',
        help='Use LAMMPS ML-IAP interface.',
        action='store_true',
    )
    for name, help_text in (
        ('cuda_graph', 'Replay the model on one CUDA MPI rank.'),
        ('ghost_folding', 'Fold ghost atoms onto local owners on one CUDA rank.'),
        ('fused', 'Use fused LunaTP kernels.'),
        ('radial_table', 'Interpolate radial weights with cubic Hermite tables.'),
    ):
        flags = ag.add_mutually_exclusive_group()
        flags.add_argument(f'--{name}', action='store_true', help=help_text)
        flags.add_argument(
            f'--disable_{name}', dest=name, action='store_false',
            help=f'Disable {name} for ML-IAP.',
        )
        ag.set_defaults(**{name: None})
    ag.add_argument(
        '--radial_knots', type=int,
        help=f'Number of Hermite knots (default: {DEFAULT_RADIAL_TABLE_KNOTS}).',
    )
    ag.add_argument(
        '--radial_r_min', type=float,
        help=f'First knot in angstroms (default: {DEFAULT_RADIAL_TABLE_R_MIN:.2f}).',
    )


def run(args):
    import sevenn.util

    checkpoint = args.checkpoint
    output_prefix = args.output_prefix
    get_parallel = args.get_parallel
    get_serial = not get_parallel
    modal = args.modal
    use_flash = args.enable_flash
    use_cueq = args.enable_cueq
    use_oeq = args.enable_oeq
    use_lunatp = args.enable_lunatp
    use_mliap = args.use_mliap
    cuda_graph = args.cuda_graph
    ghost_folding = args.ghost_folding
    radial_table = args.radial_table
    fused = args.fused
    mliap_options = any(
        value is not None for value in (
            cuda_graph, ghost_folding, fused, radial_table,
            args.radial_knots, args.radial_r_min,
        )
    )

    if args.compact_checkpoint:
        if any((
            get_parallel, modal, use_flash, use_cueq, use_oeq, use_lunatp,
            use_mliap, mliap_options,
        )):
            raise ValueError(
                '--compact_checkpoint writes a backend-independent .pth'
            )
        output = output_prefix or 'inference.pth'
        if not output.endswith('.pth'):
            output += '.pth'
        sevenn.util.load_checkpoint(checkpoint).save_inference_checkpoint(output)
        return

    # Check dependencies
    if use_flash:
        from sevenn.nn.flash_helper import is_flash_available

        if not is_flash_available():
            raise ImportError('FlashTP not installed or no GPU found.')

    if use_cueq:
        from sevenn.nn.cue_helper import is_cue_available

        if not is_cue_available():
            raise ImportError('cuEquivariance is not installed.')

    if use_oeq:
        from sevenn.nn.oeq_helper import is_oeq_available

        if not is_oeq_available():
            raise ImportError('OpenEquivariance not installed or no GPU found.')

    if use_lunatp:
        from sevenn.nn.lunatp_helper import is_lunatp_available

        if not is_lunatp_available():
            raise ImportError('LunaTP not installed or no GPU found.')

    if use_cueq and not use_mliap:
        raise ValueError('cuEquivariance is only supported in ML-IAP interface.')

    if use_lunatp and not use_mliap:
        raise ValueError('LunaTP is only supported in ML-IAP interface.')

    if mliap_options and not use_mliap:
        raise ValueError('These acceleration options require --use_mliap.')
    if fused and not use_lunatp:
        raise ValueError('fused needs LunaTP (--enable_lunatp).')

    if use_mliap and get_parallel:
        raise ValueError(
            'Deploy ML-IAP without --get_parallel. '
            'It uses the same potential for single- and multi-rank runs.'
        )

    # deploy
    if output_prefix is None:
        output_prefix = 'deployed_parallel' if not get_serial else 'deployed_serial'

        if use_mliap:
            output_prefix += '_mliap'

    checkpoint_path = None
    if os.path.isfile(checkpoint):
        checkpoint_path = checkpoint
    else:
        checkpoint_path = sevenn.util.pretrained_name_to_path(checkpoint)

    if not use_mliap:
        from sevenn.scripts.deploy import deploy, deploy_parallel

        if get_serial:
            deploy(checkpoint_path, output_prefix, modal, use_flash=use_flash, use_oeq=use_oeq)  # noqa: E501
        else:
            deploy_parallel(checkpoint_path, output_prefix, modal, use_flash=use_flash, use_oeq=use_oeq)  # noqa: E501
    else:
        from sevenn import mliap

        if not (use_cueq or use_flash or use_oeq or use_lunatp):
            sevenn.util.warn_no_tp_accelerator(
                'LAMMPS ML-IAP deployment',
            )

        if output_prefix.endswith('.pt') is False:
            output_prefix += '.pt'

        mliap_module = mliap.SevenNetMLIAPWrapper(
            model_path=checkpoint,
            modal=modal,
            use_cueq=use_cueq,
            use_flash=use_flash,
            use_oeq=use_oeq,
            use_lunatp=use_lunatp,
            use_cuda_graph=cuda_graph,
            use_ghost_folding=ghost_folding,
            use_radial_table=radial_table,
            use_fused=fused,
            radial_knots=args.radial_knots,
            radial_r_min=args.radial_r_min,
        )
        torch.save(mliap_module, output_prefix)


# legacy way
def main():
    ag = argparse.ArgumentParser(description=description_get_model)
    add_args(ag)
    run(ag.parse_args())
