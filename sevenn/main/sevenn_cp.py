import argparse
import os.path as osp

import yaml

from sevenn import __version__
from sevenn.util import load_checkpoint

description = (
    f'sevenn version={__version__}, sevenn_cp.\n'
    + 'tool box for checkpoints generated from sevennet'
)


def main(args=None):
    args = cmd_parse_data(args)
    checkpoint = load_checkpoint(args.checkpoint)
    if args.get_yaml:
        mode = args.get_yaml
        cfg = checkpoint.yaml_dict(mode)
        print(yaml.dump(cfg, indent=4, sort_keys=False, default_flow_style=False))
    elif args.append_modal:
        ref_yaml = args.append_modal
        if not osp.exists(ref_yaml):
            raise FileNotFoundError(f'No yaml file {ref_yaml}')
        with open(ref_yaml, 'r') as f:
            ref_config = yaml.safe_load(f)  # noqa


def cmd_parse_data(args=None):
    ag = argparse.ArgumentParser(description=description)

    ag.add_argument('checkpoint', help='checkpoint or pretrained', type=str)

    group = ag.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--get_yaml',
        choices=['reproduce', 'continue', 'continue_modal'],
        help='create input.yaml based on the given checkpoint',
        type=str,
    )

    group.add_argument(
        '--append_modal',
        help='append modality with given yaml. Yaml file changes in-place.',
        type=str,
    )

    args = ag.parse_args()
    return args
