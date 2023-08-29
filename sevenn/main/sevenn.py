import os
import sys
import argparse

from sevenn.parse_input import read_config_yaml
from sevenn.scripts.train import train
from sevenn.sevenn_logger import Logger
from sevenn._const import SEVENN_VERSION
import sevenn._keys as KEY

description = f"sevenn version={SEVENN_VERSION}, based on input.yaml, train model"

input_yaml_help = "main input.yaml file for model & training"
working_dir_help = "directory path to write output. default is cwd"
screen_help = "print log output to screen or not"
get_parallel_help = "deploy parallel model from serial one"

# TODO: do somthing for model type (it is not printed on log)
global_config = {'version': SEVENN_VERSION,
                 KEY.MODEL_TYPE: 'E3_equivariant_model'}


def main(args=None):
    """
    main function of sevenn
    """
    input_yaml, working_dir, screen = cmd_parse_main(args)
    if working_dir is None:
        working_dir = os.getcwd()
    Logger(filename=f"{os.path.abspath(working_dir)}/log.sevenn", screen=screen)
    Logger().greeting()

    try:
        model_config, train_config, data_config = read_config_yaml(input_yaml)
    except Exception as e:
        Logger().error(e)
        sys.exit(1)

    Logger().print_config(model_config, data_config, train_config)
    # don't have to distinguish configs inside program
    global_config.update(model_config)
    global_config.update(train_config)
    global_config.update(data_config)

    # Not implemented
    if global_config[KEY.DTYPE] == "double":
        raise Exception("double precision is not implemented yet")
        #torch.set_default_dtype(torch.double)

    # run train
    train(global_config, working_dir)


def cmd_parse_main(args=None):
    ag = argparse.ArgumentParser(description=description)
    ag.add_argument('input_yaml', help=input_yaml_help, type=str)
    ag.add_argument('-w', '--working_dir', nargs='?', const=os.getcwd(),
                    help=working_dir_help, type=str)
    ag.add_argument('-s', '--screen', help=screen_help, action='store_true')

    args = ag.parse_args()
    input_yaml = args.input_yaml
    wd = args.working_dir
    return input_yaml, wd, args.screen


if __name__ == "__main__":
    main()

