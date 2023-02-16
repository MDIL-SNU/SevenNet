import sys
import traceback
from datetime import datetime
"""
You don't have to understand what metaclass is.
Understanding what singletone is enough
If you're interested see best answer of
https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
"""

from sevenn.train.trainer import DataSetType
import sevenn._const as _const
import sevenn._keys as KEY


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    """
    logger for simple gnn
    """
    SCREEN_WIDTH = 104  # from vasp OUTCAR

    def __init__(self, filename: str, screen: bool):
        self.logfile = open(filename, 'w', buffering=1)
        self.screen = screen
        self.timer_dct = {}

    def __del__(self):
        self.logfile.close()

    def write(self, content: str):
        # no newline!
        self.logfile.write(content)
        if self.screen:
            print(content, end='')

    def epoch_write(self, loss_history, idx=-1):
        lb_pad = 29
        fs = 9
        pad = 29 - fs
        content = \
            f"{'Label':{lb_pad}}{'E_RMSE(T)':<{pad}}{'F_RMSE(T)':<{pad}}".\
            format(lb_pad=lb_pad, pad=pad)\
            + f"{'E_RMSE(V)':<{pad}}{'F_RMSE(V)':<{pad}}\n".format(pad=pad)
        train_loss = loss_history[DataSetType.TRAIN]
        valid_loss = loss_history[DataSetType.VALID]
        label_keys = train_loss.keys()
        for label in label_keys:
            t_E = train_loss[label]['energy'][idx]
            t_F = train_loss[label]['force'][idx]
            v_E = valid_loss[label]['energy'][idx]
            v_F = valid_loss[label]['force'][idx]
            content += "{label:{lb_pad}}{t_E:<{pad}.{fs}f}{t_F:<{pad}.{fs}f}".\
                format(label=label, t_E=t_E, t_F=t_F, lb_pad=lb_pad, pad=pad, fs=fs)\
                + "{v_E:<{pad}.{fs}f}{v_F:<{pad}.{fs}f}\n".\
                format(v_E=v_E, v_F=v_F, pad=pad, fs=fs)
        self.write(content)

    @staticmethod
    def format_k_v(key, val, write=False):
        MAX_KEY_SIZE = 25
        SPERATOR = ', '
        EMPTY_PADDING = ' ' * (MAX_KEY_SIZE + 3)
        val = str(val)
        content = f"{key:<{MAX_KEY_SIZE}}: {val}"
        if len(content) > Logger.SCREEN_WIDTH:
            content = f"{key:<{MAX_KEY_SIZE}}: "
            # sperate val by sperator
            val_list = val.split(SPERATOR)
            current_len = len(content)
            for val_compo in val_list:
                current_len += len(val_compo)
                if current_len > Logger.SCREEN_WIDTH:
                    newline_content = f"{EMPTY_PADDING}{val_compo}{SPERATOR}"
                    content += f"\\\n{newline_content}"
                    current_len = len(newline_content)
                else:
                    content += f"{val_compo}{SPERATOR}"

        if content.endswith(f"{SPERATOR}"):
            content = content[:-len(SPERATOR)]
        content += '\n'

        if write is False:
            return content
        else:
            Logger().write(content)

    def greeting(self):
        content = f"sevenn version {_const.SEVENN_VERSION}\n"
        content += "reading yaml config..."
        self.write(f"sevenn version {_const.SEVENN_VERSION}\n")

    def bar(self):
        content = "-" * Logger.SCREEN_WIDTH + "\n"
        self.write(content)

    def print_config(self, model_config, data_config, train_config):
        """
        print some important information from config
        """
        content = "succesfully read yaml config!\n\n" + "from model configuration\n"
        for k, v in model_config.items():
            content += Logger.format_k_v(k, v)
        content += "\nfrom train configuration\n"
        for k, v in train_config.items():
            content += Logger.format_k_v(k, v)
        content += "\nfrom data configuration\n"
        for k, v in data_config.items():
            content += Logger.format_k_v(k, v)
        self.write(content)

    def error(self, e: Exception):
        content = ""
        if type(e) is ValueError:
            content += "Error occured!\n"
            content += str(e) + "\n"
        else:
            content += "Unknown error occured!\n"
            content += traceback.format_exc()
        self.write(content)

    def timer_start(self, name: str):
        self.timer_dct[name] = datetime.now()

    def timer_end(self, name: str, message: str, remove=True):
        """
        print f"{message}: {elapsed}"
        """
        elapsed = str(datetime.now() - self.timer_dct[name])
        # elapsed = elapsed.strftime('%H-%M-%S')
        del self.timer_dct[name]
        self.write(f"{message}: {elapsed[:-4]}\n")


