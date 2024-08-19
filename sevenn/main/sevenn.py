import argparse
import os
import sys

import torch.distributed as dist

import sevenn._keys as KEY
from sevenn import __version__
from sevenn.parse_input import read_config_yaml
from sevenn.scripts.train import train
from sevenn.sevenn_logger import Logger

description = (
    f'sevenn version={__version__}, train model based on the input.yaml'
)

input_yaml_help = 'input.yaml for training'
working_dir_help = 'path to write output. Default is cwd.'
screen_help = 'print log to stdout'
distributed_help = 'set this flag if it is distributed training'

# TODO: do something for model type (it is not printed on log)
global_config = {
    'version': __version__,
    KEY.MODEL_TYPE: 'E3_equivariant_model',
}


def main(args=None):
    """
    main function of sevenn
    """
    input_yaml, working_dir, screen, distributed = cmd_parse_main(args)

    if working_dir is None:
        working_dir = os.getcwd()

    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(
            backend='nccl', world_size=world_size, rank=rank
        )
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    Logger(
        filename=f'{os.path.abspath(working_dir)}/log.sevenn',
        screen=screen,
        rank=rank,
    )
    Logger().greeting()

    if distributed:
        Logger().writeline(
            f'Distributed training enabled, total world size is {world_size}'
        )

    try:
        model_config, train_config, data_config = read_config_yaml(input_yaml)
    except Exception as e:
        Logger().error(e)
        sys.exit(1)

    train_config[KEY.IS_DDP] = distributed
    train_config[KEY.LOCAL_RANK] = local_rank
    train_config[KEY.RANK] = rank
    train_config[KEY.WORLD_SIZE] = world_size

    Logger().print_config(model_config, data_config, train_config)
    # don't have to distinguish configs inside program
    global_config.update(model_config)
    global_config.update(train_config)
    global_config.update(data_config)

    # Not implemented
    if global_config[KEY.DTYPE] == 'double':
        raise Exception('double precision is not implemented yet')
        # torch.set_default_dtype(torch.double)

    # run train
    train(global_config, working_dir)


def cmd_parse_main(args=None):
    ag = argparse.ArgumentParser(description=description)
    ag.add_argument('input_yaml', help=input_yaml_help, type=str)
    ag.add_argument(
        '-w',
        '--working_dir',
        nargs='?',
        const=os.getcwd(),
        help=working_dir_help,
        type=str,
    )
    ag.add_argument('-s', '--screen', help=screen_help, action='store_true')
    ag.add_argument(
        '-d', '--distributed', help=distributed_help, action='store_true'
    )

    args = ag.parse_args()
    input_yaml = args.input_yaml
    wd = args.working_dir
    return input_yaml, wd, args.screen, args.distributed


if __name__ == '__main__':
    main()
