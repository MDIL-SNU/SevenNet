import argparse
import os.path as osp

from sevenn import __version__

description = (
    'tool box for sevennet checkpoints'
)


def add_parser(subparsers):
    ag = subparsers.add_parser('checkpoint', help=description, aliases=['cp'])
    add_args(ag)


def add_args(parser):
    ag = parser

    ag.add_argument('checkpoint', help='checkpoint or pretrained', type=str)

    group = ag.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--get_yaml',
        choices=['reproduce', 'continue', 'continue_modal'],
        help='create input.yaml based on the given checkpoint',
        type=str,
    )

    group.add_argument(
        '--append_modal_yaml',
        help='append modality with given yaml.',
        type=str,
    )
    ag.add_argument(
        '--original_modal_name',
        help=(
            'when the append_modal is used and checkpoint is not multi-modal, '
            + 'used to name previously trained modality. defaults to "origin"'
        ),
        default='origin',
        type=str,
    )


def run(args):
    import torch
    import yaml

    from sevenn.parse_input import read_config_yaml
    from sevenn.util import load_checkpoint

    checkpoint = load_checkpoint(args.checkpoint)
    if args.get_yaml:
        mode = args.get_yaml
        cfg = checkpoint.yaml_dict(mode)
        print(yaml.dump(cfg, indent=4, sort_keys=False, default_flow_style=False))
    elif args.append_modal_yaml:
        dst_yaml = args.append_modal_yaml
        if not osp.exists(dst_yaml):
            raise FileNotFoundError(f'No yaml file {dst_yaml}')

        dst_config = read_config_yaml(dst_yaml, return_separately=False)
        model_state_dict = checkpoint.append_modal(
            dst_config, args.original_modal_name
        )

        to_save = checkpoint.get_checkpoint_dict()
        to_save.update({'config': dst_config, 'model_state_dict': model_state_dict})

        torch.save(to_save, 'checkpoint_modal_appended.pth')
        print('checkpoint_modal_appended.pth is successfully saved.')
        print(f'update continue of {dst_yaml} as blow (recommend) to continue')
        cont_dct = {
            'continue': {
                'checkpoint': 'checkpoint_modal_appended.pth',
                'reset_epoch': True,
                'reset_optimizer': True,
                'reset_scheduler': True,
            }
        }
        print(
            yaml.dump(cont_dct, indent=4, sort_keys=False, default_flow_style=False)
        )

    else:
        print(checkpoint)


def main(args=None):
    ag = argparse.ArgumentParser(description=description)
    add_args(ag)
    run(ag.parse_args())
