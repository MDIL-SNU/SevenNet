import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from ase.data import atomic_numbers

import sevenn._keys as KEY
from sevenn import __version__

CHEM_SYMBOLS = {v: k for k, v in atomic_numbers.items()}


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    SCREEN_WIDTH = 120  # half size of my screen / changed due to stress output

    def __init__(
        self, filename: Optional[str] = None, screen: bool = False, rank: int = 0
    ):
        self.rank = rank
        self._filename = filename
        if rank == 0:
            # if filename is not None:
            #    self.logfile = open(filename, 'a', buffering=1)
            self.logfile = None
            self.files = {}
            self.screen = screen
        else:
            self.logfile = None
            self.screen = False
        self.timer_dct = {}
        self.active = True

    def __enter__(self):
        if self.rank != 0:
            return self
        if self.logfile is None and self._filename is not None:
            try:
                self.logfile = open(
                    self._filename, 'a', buffering=1, encoding='utf-8'
                )
            except IOError as e:
                print(f'Failed to re-open log file {self._filename}: {e}')
                self.logfile = None
            self.files = {}
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.rank != 0:
            return self
        try:
            if self.logfile is not None:
                self.logfile.close()
                self.logfile = None
            for f in self.files.values():
                f.close()
        except IOError as e:
            print(f'Failed to close log files: {e}')
        finally:
            self.logfile = None
            self.files = {}

    def switch_file(self, new_filename: str):
        if self.rank != 0:
            return self
        if self.logfile is not None:
            raise ValueError('Current logfile is not yet closed')
        self._filename = new_filename
        return self

    def write(self, content: str):
        if self.rank != 0:
            return
        # no newline!
        if self.logfile is not None and self.active:
            self.logfile.write(content)
        if self.screen and self.active:
            print(content, end='')

    def writeline(self, content: str):
        content = content + '\n'
        self.write(content)

    def init_csv(self, filename: str, header: list):
        """
        Deprecated
        """
        if self.rank == 0:
            self.files[filename] = open(filename, 'w', buffering=1, encoding='utf-8')
            self.files[filename].write(','.join(header) + '\n')
        else:
            pass

    def append_csv(self, filename: str, content: list, decimal: int = 6):
        """
        Deprecated
        """
        if self.rank == 0:
            if filename not in self.files:
                self.files[filename] = open(filename, 'a', buffering=1)
            str_content = []
            for c in content:
                if isinstance(c, float):
                    str_content.append(f'{c:.{decimal}f}')
                else:
                    str_content.append(str(c))
            self.files[filename].write(','.join(str_content) + '\n')
        else:
            pass

    def natoms_write(self, natoms: Dict[str, Dict]):
        content = ''
        total_natom = {}
        for label, natom in natoms.items():
            content += self.format_k_v(label, natom)
            for specie, num in natom.items():
                try:
                    total_natom[specie] += num
                except KeyError:
                    total_natom[specie] = num
        content += self.format_k_v('Total, label wise', total_natom)
        content += self.format_k_v('Total', sum(total_natom.values()))
        self.write(content)

    def statistic_write(self, statistic: Dict[str, Dict]):
        content = ''
        for label, dct in statistic.items():
            if label.startswith('_'):
                continue
            if not isinstance(dct, dict):
                continue
            dct_new = {}
            for k, v in dct.items():
                if k.startswith('_'):
                    continue
                if isinstance(v, int):
                    dct_new[k] = v
                else:
                    dct_new[k] = f'{v:.3f}'
            content += self.format_k_v(label, dct_new)
        self.write(content)

    # TODO : refactoring!!!, this is not loss, rmse
    def epoch_write_specie_wise_loss(self, train_loss, valid_loss):
        lb_pad = 21
        fs = 6
        pad = 21 - fs
        ln = '-' * fs
        total_atom_type = train_loss.keys()
        content = ''

        for at in total_atom_type:
            t_F = train_loss[at]
            v_F = valid_loss[at]
            at_sym = CHEM_SYMBOLS[at]
            content += '{label:{lb_pad}}{t_E:<{pad}.{fs}s}{v_E:<{pad}.{fs}s}'.format(
                label=at_sym, t_E=ln, v_E=ln, lb_pad=lb_pad, pad=pad, fs=fs
            ) + '{t_F:<{pad}.{fs}f}{v_F:<{pad}.{fs}f}'.format(
                t_F=t_F, v_F=v_F, pad=pad, fs=fs
            )
            content += '{t_S:<{pad}.{fs}s}{v_S:<{pad}.{fs}s}'.format(
                t_S=ln, v_S=ln, pad=pad, fs=fs
            )
            content += '\n'
        self.write(content)

    def write_full_table(
        self,
        dict_list: List[Dict],
        row_labels: List[str],
        decimal_places: int = 6,
        pad: int = 2,
    ):
        """
        Assume data_list is list of dict with same keys
        """
        assert len(dict_list) == len(row_labels)
        label_len = max(map(len, row_labels))
        # Extract the column names and create a 2D array of values
        col_names = list(dict_list[0].keys())

        values = [list(d.values()) for d in dict_list]

        # Format the numbers with the given decimal places
        formatted_values = [
            [f'{value:.{decimal_places}f}' for value in row] for row in values
        ]

        # Calculate padding lengths for each column (with extra padding)
        max_col_lengths = [
            max(len(str(value)) for value in col) + pad
            for col in zip(col_names, *formatted_values)
        ]

        # Create header row and separator
        header = ' ' * (label_len + pad) + ' '.join(
            col_name.ljust(pad) for col_name, pad in zip(col_names, max_col_lengths)
        )
        separator = '-'.join('-' * pad for pad in max_col_lengths) + '-' * (
            label_len + pad
        )

        # Print header and separator
        self.writeline(header)
        self.writeline(separator)

        # Print the data rows with row labels
        for row_label, row in zip(row_labels, formatted_values):
            data_row = ' '.join(
                value.rjust(pad) for value, pad in zip(row, max_col_lengths)
            )
            self.writeline(f'{row_label.ljust(label_len)}{data_row}')

    def format_k_v(self, key: Any, val: Any, write: bool = False):
        """
        key and val should be str convertible
        """
        MAX_KEY_SIZE = 20
        SEPARATOR = ', '
        EMPTY_PADDING = ' ' * (MAX_KEY_SIZE + 3)
        NEW_LINE_LEN = Logger.SCREEN_WIDTH - 5
        key = str(key)
        val = str(val)
        content = f'{key:<{MAX_KEY_SIZE}}: {val}'
        if len(content) > NEW_LINE_LEN:
            content = f'{key:<{MAX_KEY_SIZE}}: '
            # septate val by separator
            val_list = val.split(SEPARATOR)
            current_len = len(content)
            for val_compo in val_list:
                current_len += len(val_compo)
                if current_len > NEW_LINE_LEN:
                    newline_content = f'{EMPTY_PADDING}{val_compo}{SEPARATOR}'
                    content += f'\\\n{newline_content}'
                    current_len = len(newline_content)
                else:
                    content += f'{val_compo}{SEPARATOR}'

        if content.endswith(f'{SEPARATOR}'):
            content = content[: -len(SEPARATOR)]
        content += '\n'

        if write is False:
            return content
        else:
            self.write(content)
            return ''

    def greeting(self):
        LOGO_ASCII_FILE = f'{os.path.dirname(__file__)}/logo_ascii'
        with open(LOGO_ASCII_FILE, 'r') as logo_f:
            logo_ascii = logo_f.read()
        content = 'SevenNet: Scalable EquVariance-Enabled Neural Network\n'
        content += f'version {__version__}, {time.ctime()}\n'
        content += f'this file: {self._filename}\n'
        content += 'reading yaml config...'
        self.write(content)
        self.write(logo_ascii)

    def bar(self):
        content = '-' * Logger.SCREEN_WIDTH + '\n'
        self.write(content)

    def print_config(
        self,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any],
        train_config: Dict[str, Any],
    ):
        """
        print some important information from config
        """
        content = 'successfully read yaml config!\n\n' + 'from model configuration\n'
        for k, v in model_config.items():
            content += self.format_k_v(k, str(v))
        content += '\nfrom train configuration\n'
        for k, v in train_config.items():
            content += self.format_k_v(k, str(v))
        content += '\nfrom data configuration\n'
        for k, v in data_config.items():
            content += self.format_k_v(k, str(v))
        self.write(content)

    # TODO: This is not good make own exception
    def error(self, e: Exception):
        content = ''
        if type(e) is ValueError:
            content += 'Error occurred!\n'
            content += str(e) + '\n'
        else:
            content += 'Unknown error occurred!\n'
            content += traceback.format_exc()
        self.write(content)

    def timer_start(self, name: str):
        self.timer_dct[name] = datetime.now()

    def timer_end(self, name: str, message: str, remove: bool = True):
        """
        print f"{message}: {elapsed}"
        """
        elapsed = str(datetime.now() - self.timer_dct[name])
        # elapsed = elapsed.strftime('%H-%M-%S')
        if remove:
            del self.timer_dct[name]
        self.write(f'{message}: {elapsed[:-4]}\n')

    # TODO: print it without config
    # TODO: refactoring, readout part name :(
    def print_model_info(self, model, config):
        from functools import partial

        kv_write = partial(self.format_k_v, write=True)
        self.writeline('Irreps of features')
        kv_write('edge_feature', model.get_irreps_in('edge_embedding', 'irreps_out'))
        for i in range(config[KEY.NUM_CONVOLUTION]):
            kv_write(
                f'{i}th node',
                model.get_irreps_in(f'{i}_self_interaction_1'),
            )
        i = config[KEY.NUM_CONVOLUTION] - 1
        kv_write(
            'readout irreps',
            model.get_irreps_in(f'{i}_equivariant_gate', 'irreps_out'),
        )

        num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.writeline(f'# learnable parameters: {num_weights}\n')
