import argparse
import os

from sevenn import __version__

description_get_model = (
    'deploy LAMMPS model from the checkpoint'
)
checkpoint_help = (
    'path to the checkpoint | SevenNet-0 | 7net-0 |'
    ' {SevenNet-0|7net-0}_{11July2024|22May2024}'
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
        '--enable_flashTP',
        help='use flashTP. LAMMPS must be specially compiled.',
        action='store_true',
    )


def run(args):
    import sevenn.util
    from sevenn.scripts.deploy import deploy, deploy_parallel

    checkpoint = args.checkpoint
    output_prefix = args.output_prefix
    get_parallel = args.get_parallel
    get_serial = not get_parallel
    modal = args.modal
    use_flash = args.enable_flashTP

    if output_prefix is None:
        output_prefix = 'deployed_parallel' if not get_serial else 'deployed_serial'

    checkpoint_path = None
    if os.path.isfile(checkpoint):
        checkpoint_path = checkpoint
    else:
        checkpoint_path = sevenn.util.pretrained_name_to_path(checkpoint)

    if use_flash:
        import sevenn.nn.flash_helper

        if not sevenn.nn.flash_helper.is_flash_available():
            raise ImportError('FlashTP not installed or no GPU found.')

    if get_serial:
        deploy(checkpoint_path, output_prefix, modal, use_flash=use_flash)
    else:
        deploy_parallel(checkpoint_path, output_prefix, modal, use_flash=use_flash)


# legacy way
def main():
    ag = argparse.ArgumentParser(description=description_get_model)
    add_args(ag)
    run(ag.parse_args())
