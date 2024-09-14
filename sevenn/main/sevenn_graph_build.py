import argparse
import os
from datetime import datetime

import sevenn.scripts.graph_build as graph_build
from sevenn import __version__
from sevenn.sevenn_logger import Logger

description = (
    f'sevenn version={__version__}, sevenn_graph_build.\n'
    + "Create '.sevenn_data' from ase readable or VASP OUTCARs (by"
    ' structure_list).\n'
)

source_help = 'source data to build graph'
label_by_help = 'label the output dataset with the given string.'
cutoff_help = 'cutoff radius of edges in Angstrom'
suffix_help = 'when source is dir, suffix of the files.'
copy_info_help = 'copy ase.Atoms.info to output dataset'
format_help = (
    'type of the source, default is structure_list. '
    + 'Otherwise, it is directly passed to ase.io.read'
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
        'sevenn_version': __version__,
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
        help='path to write outputs',
        type=str,
        default='./',
    )
    ag.add_argument(
        '-sb',
        '--save_by_label',
        help=(
            'if source is structure_list, separate the output dataset by label'
        ),
        action='store_true',
        default=False,
    )
    ag.add_argument(
        '--kwargs',
        nargs=argparse.REMAINDER,
        help='will be passed to ase.io.read, or can be used to specify EFS key',
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
