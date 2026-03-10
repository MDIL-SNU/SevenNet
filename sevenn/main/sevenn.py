import argparse
import os
import sys
import time

from sevenn import __version__

description = 'train a model given the input.yaml'

input_yaml_help = 'input.yaml for training'
mode_help = 'main training script to run. Default is train.'
working_dir_help = 'path to write output. Default is cwd.'
screen_help = 'print log to stdout'
distributed_help = 'set this flag if it is distributed training'
distributed_backend_help = 'backend for distributed training. Supported: nccl, mpi'

# Metainfo will be saved to checkpoint
global_config = {
    'version': __version__,
    'when': time.ctime(),
    '_model_type': 'E3_equivariant_model',
}


def run(args):
    """
    main function of sevenn
    """
    import random
    import sys

    import torch
    import torch.distributed as dist

    import sevenn._keys as KEY
    from sevenn.logger import Logger
    from sevenn.parse_input import read_config_yaml
    from sevenn.scripts.train import train, train_v2
    from sevenn.util import unique_filepath

    input_yaml = args.input_yaml
    mode = args.mode
    working_dir = args.working_dir
    log = args.log
    screen = args.screen
    distributed = args.distributed
    distributed_backend = args.distributed_backend
    use_cue = args.enable_cueq
    use_flash = args.enable_flash
    use_oeq = args.enable_oeq

    if use_cue:
        import sevenn.nn.cue_helper

        if not sevenn.nn.cue_helper.is_cue_available():
            raise ImportError('cuEquivariance not installed or no GPU found.')

    if use_flash:
        import sevenn.nn.flash_helper

        if not sevenn.nn.flash_helper.is_flash_available():
            raise ImportError('FlashTP not installed or no GPU found.')

    if use_oeq:
        import sevenn.nn.oeq_helper

        if not sevenn.nn.oeq_helper.is_oeq_available():
            raise ImportError('OpenEquivariance not installed or no GPU found.')

    if working_dir is None:
        working_dir = os.getcwd()
    elif not os.path.isdir(working_dir):
        os.makedirs(working_dir, exist_ok=True)

    world_size = 1
    if distributed:
        if distributed_backend == 'nccl':
            local_rank = int(os.environ['LOCAL_RANK'])
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        elif distributed_backend == 'mpi':
            local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        else:
            raise ValueError(f'Unknown distributed backend: {distributed_backend}')

        dist.init_process_group(
            backend=distributed_backend, world_size=world_size, rank=rank
        )
    else:
        local_rank, rank, world_size = 0, 0, 1

    log_fname = unique_filepath(f'{os.path.abspath(working_dir)}/{log}')
    with Logger(filename=log_fname, screen=screen, rank=rank) as logger:
        logger.greeting()

        if distributed:
            logger.writeline(
                f'Distributed training enabled, total world size is {world_size}'
            )

        try:
            model_config, train_config, data_config = read_config_yaml(input_yaml)
        except Exception as e:
            logger.writeline('Failed to parsing input.yaml')
            logger.error(e)
            sys.exit(1)

        train_config[KEY.IS_DDP] = distributed
        train_config[KEY.DDP_BACKEND] = distributed_backend
        train_config[KEY.LOCAL_RANK] = local_rank
        train_config[KEY.RANK] = rank
        train_config[KEY.WORLD_SIZE] = world_size

        if distributed:
            torch.cuda.set_device(torch.device('cuda', local_rank))

        if use_cue:
            if KEY.CUEQUIVARIANCE_CONFIG not in model_config:
                model_config[KEY.CUEQUIVARIANCE_CONFIG] = {'use': True}
            else:
                model_config[KEY.CUEQUIVARIANCE_CONFIG].update({'use': True})

        if use_flash:
            model_config[KEY.USE_FLASH_TP] = True

        if use_oeq:
            model_config[KEY.USE_OEQ] = True

        logger.print_config(model_config, data_config, train_config)
        # don't have to distinguish configs inside program
        global_config.update(model_config)
        global_config.update(train_config)
        global_config.update(data_config)

        # Not implemented
        if global_config[KEY.DTYPE] == 'double':
            raise Exception('double precision is not implemented yet')
            # torch.set_default_dtype(torch.double)

        seed = global_config[KEY.RANDOM_SEED]
        random.seed(seed)
        torch.manual_seed(seed)

        # run train
        if mode == 'train_v1':
            train(global_config, working_dir)
        elif mode == 'train_v2':
            train_v2(global_config, working_dir)


def cmd_parser_train(parser):
    ag = parser
    ag.add_argument('input_yaml', help=input_yaml_help, type=str)
    ag.add_argument(
        '-m',
        '--mode',
        choices=['train_v1', 'train_v2'],
        default='train_v2',
        help=mode_help,
        type=str,
    )
    ag.add_argument(
        '-cueq',
        '--enable_cueq',
        help='use cuEq accelerations for training',
        action='store_true',
    )
    ag.add_argument(
        '-flashTP',
        '--enable_flashTP',
        '--enable_flash',
        dest='enable_flash',
        help='use flashTP accelerations for training',
        action='store_true',
    )
    ag.add_argument(
        '-oeq',
        '--enable_oeq',
        help='use OpenEquivariance accelerations for training',
        action='store_true',
    )
    ag.add_argument(
        '-w',
        '--working_dir',
        nargs='?',
        const=os.getcwd(),
        help=working_dir_help,
        type=str,
    )
    ag.add_argument(
        '-l',
        '--log',
        default='log.sevenn',
        help='name of logfile, default is log.sevenn',
        type=str,
    )
    ag.add_argument('-s', '--screen', help=screen_help, action='store_true')
    ag.add_argument(
        '-d', '--distributed', help=distributed_help, action='store_true'
    )
    ag.add_argument(
        '--distributed_backend',
        help=distributed_backend_help,
        type=str,
        default='nccl',
        choices=['nccl', 'mpi'],
    )


def add_parser(subparsers):
    ag = subparsers.add_parser('train', help=description)
    cmd_parser_train(ag)


def set_default_subparser(self, name):
    """default subparser selection. Call after setup, just before parse_args()
    name: is the name of the subparser to call by default
    args: if set is the argument list handed to parse_args()

    Hack copied from stack overflow
    """

    subparser_found = False
    for arg in sys.argv[1:]:
        if arg in ['-h', '--help']:  # global help if no subparser
            break
    else:
        for x in self._subparsers._actions:
            if not isinstance(x, argparse._SubParsersAction):
                continue
            for sp_name in x._name_parser_map.keys():
                if sp_name in sys.argv[1:]:
                    subparser_found = True
        if not subparser_found:
            # we don't have global option except -h. So simply put 'train' to 1
            sys.argv.insert(1, name)


argparse.ArgumentParser.set_default_subparser = set_default_subparser  # type: ignore


def main():
    import sevenn.main.sevenn_cp as checkpoint_cmd
    import sevenn.main.sevenn_get_model as get_model_cmd
    import sevenn.main.sevenn_graph_build as graph_build_cmd
    import sevenn.main.sevenn_inference as inference_cmd
    import sevenn.main.sevenn_patch_lammps as patch_lammps_cmd
    import sevenn.main.sevenn_preset as preset_cmd

    ag = argparse.ArgumentParser(f'SevenNet version={__version__}')

    subparsers = ag.add_subparsers(dest='command', help='Sub-commands')
    add_parser(subparsers)  # add 'train'
    checkpoint_cmd.add_parser(subparsers)
    inference_cmd.add_parser(subparsers)
    graph_build_cmd.add_parser(subparsers)
    preset_cmd.add_parser(subparsers)
    get_model_cmd.add_parser(subparsers)
    patch_lammps_cmd.add_parser(subparsers)

    ag.set_default_subparser('train')  # type: ignore
    args = ag.parse_args()

    if args.command == 'train':
        run(args)
    elif args.command in ['checkpoint', 'cp']:
        checkpoint_cmd.run(args)
    elif args.command in ['get_model', 'deploy']:
        get_model_cmd.run(args)
    elif args.command == 'graph_build':
        graph_build_cmd.run(args)
    elif args.command in ['inference', 'inf']:
        inference_cmd.run(args)
    elif args.command == 'patch_lammps':
        patch_lammps_cmd.run(args)
    elif args.command == 'preset':
        preset_cmd.run(args)


if __name__ == '__main__':
    main()
