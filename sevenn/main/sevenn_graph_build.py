import argparse
import os
import sys
from datetime import datetime

import sevenn._keys as KEY
import sevenn.scripts.graph_build as graph_build
from sevenn._const import SEVENN_VERSION
from sevenn.sevenn_logger import Logger

description = (
    f'sevenn version={SEVENN_VERSION}, sevenn_graph_build.\n'
    + 'Note that this command is optional. You can build graph by writting'
    ' appropriate input.yaml for training.\n'
    + "Create '.sevenn_data' from ase readable or VASP OUTCARs (by"
    " structure_list).\n"
    + 'It expects units read from ase atoms have correct units.\n'
)

source_help = (
    'Primitive data to build graph, assume structure_list if format is not'
    ' given '
    + 'or assume root dir of pickles of ase.Atoms list if directory'
)
label_by_help = (
    'label the given dataset with given string, '
    + 'if the src is dir of .pkl, default is the name of the directory '
    + 'if the src is structure_list, default is the label of the'
    ' structure_list file'
)
cutoff_help = 'cutoff radius for graph building in Angstrom '
suffix_help = (
    'when source is dir, suffix of files. if not given, read all recursively'
    + 'ignored if source is structure_list'
)
copy_info_help = (
    'do not copy ase.Atoms.info to graph data, '
    + 'ignored if source is sturcture_list'
)
format_help = (
    'format of the source, defualt is structure_list '
    + 'if it is pkl/pickle, assume they are list of ase.Atoms, '
    + 'else, the input is directly passed to ase.io.read'
)


def main(args=None):
    metadata = {}
    now = datetime.now().strftime('%Y-%m-%d')
    (
        source,
        cutoff,
        num_cores,
        label_by,
        out,
        save_by_label,
        fmt,
        suffix,
        copy_info,
        fmt_kwargs,
    ) = cmd_parse_data(args)
    metadata = {
        'sevenn_version': SEVENN_VERSION,
        'when': now,
        'cutoff': cutoff,
    }
    Logger('graph_build_log', screen=True)
    Logger().writeline(description)
    if not os.path.exists(source):
        raise ValueError(f'source {source} does not exist')

    graph_build.build_script(
        source,
        cutoff,
        num_cores,
        label_by,
        out,
        save_by_label,
        fmt,
        suffix,
        copy_info,
        metadata,
        fmt_kwargs,
    )


def cmd_parse_data(args=None):
    ag = argparse.ArgumentParser(description=description)

    ag.add_argument('source', help=source_help, type=str)
    ag.add_argument('cutoff', help=cutoff_help, type=float)
    ag.add_argument(
        '-n',
        '--num_cores',
        help='number of cores to build graph in parallel',
        default=1,
        type=int,
    )
    ag.add_argument(
        '-l', '--label_by', help=label_by_help, default='auto', type=str
    )
    ag.add_argument(
        '-f', '--format', help=format_help, type=str, default='structure_list'
    )
    ag.add_argument('-s', '--suffix', help=suffix_help, type=str, default=None)
    ag.add_argument(
        '-nc',
        '--no_copy_info',
        help=copy_info_help,
        action='store_true',
        default=False,
    )
    ag.add_argument(
        '-o',
        '--out',
        help='filename or dir path to write outputs',
        type=str,
        default='./',
    )
    ag.add_argument(
        '-sb',
        '--save_by_label',
        help='save the graph by label',
        action='store_true',
        default=False,
    )
    ag.add_argument(
        '--kwargs',
        nargs=argparse.REMAINDER,
        help='kwargs to pass to file reader (format)',
    )

    args = ag.parse_args()
    source = args.source
    cutoff = args.cutoff
    num_cores = args.num_cores
    label_by = args.label_by
    out = args.out
    fmt = args.format
    suffix = args.suffix
    save_by_label = args.save_by_label
    copy_info = not args.no_copy_info

    fmt_kwargs = {}
    if args.kwargs:
        for kwarg in args.kwargs:
            k, v = kwarg.split('=')
            fmt_kwargs[k] = v

    return (
        source,
        cutoff,
        num_cores,
        label_by,
        out,
        save_by_label,
        fmt,
        suffix,
        copy_info,
        fmt_kwargs,
    )
