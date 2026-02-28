import argparse
import os

import torch

from sevenn import __version__

description_get_model = (
    'deploy LAMMPS model from the checkpoint'
)
checkpoint_help = (
    'Pretrained model name (7net-omni, 7net-omni-i8, 7net-omni-i12, etc.) '
    'or path to checkpoint file. See documentation for all available models.'
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
        help='use OpenEquivariance. Only support ML-IAP interface.',
        action='store_true',
    )
    ag.add_argument(
        '-mliap',
        '--use_mliap',
        help='Use LAMMPS ML-IAP interface.',
        action='store_true',
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
    use_mliap = args.use_mliap

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

    if use_cueq and not use_mliap:
        raise ValueError('cuEquivariance is only supported in ML-IAP interface.')

    if use_oeq and not use_mliap:
        raise ValueError('OpenEquivariance is only supported in ML-IAP interface.')

    if use_mliap and get_parallel:
        raise ValueError('Currently, ML-IAP interface does not tested on parallel.')

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
            deploy(checkpoint_path, output_prefix, modal, use_flash=use_flash)
        else:
            deploy_parallel(checkpoint_path, output_prefix, modal, use_flash=use_flash)  # noqa: E501
    else:
        from sevenn import mliap

        if output_prefix.endswith('.pt') is False:
            output_prefix += '.pt'

        mliap_module = mliap.SevenNetMLIAPWrapper(
            model_path=checkpoint,
            modal=modal,
            use_cueq=use_cueq,
            use_flash=use_flash,
            use_oeq=use_oeq,
        )
        torch.save(mliap_module, output_prefix)


# legacy way
def main():
    ag = argparse.ArgumentParser(description=description_get_model)
    add_args(ag)
    run(ag.parse_args())
