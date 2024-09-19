import argparse
import glob
import os
import sys
from datetime import datetime

import sevenn.scripts.graph_build as graph_build
from sevenn import __version__
from sevenn.sevenn_logger import Logger
from sevenn.util import unique_filepath

description = (
    f'sevenn version={__version__}, sevenn_graph_build.\n'
    + 'Create `processed_7net/dataset.pt` or `.sevenn_data` from '
    + 'ase readable or VASP OUTCARs (by structure_list).\n'
)

source_help = 'source data to build graph, knows *'
cutoff_help = 'cutoff radius of edges in Angstrom'
log_help = 'Name of logfile. Default is graph_build_log. It never overwrite.'
legacy_help = 'build legacy .sevenn_data'


def main(args=None):
    args = cmd_parse_data(args)
    source = glob.glob(args.source)
    cutoff = args.cutoff
    num_cores = args.num_cores
    filename = args.filename
    log = args.log
    out = os.path.dirname(args.out)
    legacy = args.legacy
    print_statistics = not args.skip_statistics
    fmt_kwargs = {}
    if args.kwargs:
        for kwarg in args.kwargs:
            k, v = kwarg.split('=')
            fmt_kwargs[k] = v

    if len(source) == 0:
        print('Source has zero len, nothing to read')
        sys.exit(0)

    to_be_written = os.path.join(out, 'processed_7net', filename)
    if os.path.isfile(to_be_written):
        print(f'File already exist: {to_be_written}')
        sys.exit(0)

    metadata = {
        'sevenn_version': __version__,
        'when': datetime.now().strftime('%Y-%m-%d'),
        'cutoff': cutoff,
    }

    log_fname = unique_filepath(f'{out}/{log}')
    with Logger(filename=log_fname, screen=True) as logger:
        logger.writeline(description)

        if not legacy:
            graph_build.build_sevennet_graph_dataset(
                source, cutoff, num_cores,
                out, filename, metadata,
                print_statistics, **fmt_kwargs
            )
        else:
            graph_build.build_script(  # build .sevenn_data
                source, cutoff, num_cores, out, metadata, **fmt_kwargs,
            )


def cmd_parse_data(args=None):
    ag = argparse.ArgumentParser(description=description)

    ag.add_argument(
        'source',
        help=source_help,
        type=str
    )
    ag.add_argument(
        'cutoff',
        help=cutoff_help,
        type=float
    )
    ag.add_argument(
        '-n',
        '--num_cores',
        help='number of cores to build graph in parallel',
        default=1,
        type=int,
    )
    ag.add_argument(
        '-l',
        '--log',
        default='graph_build_log',
        help=log_help,
        type=str,
    )
    ag.add_argument(
        '-o',
        '--out',
        help='Path to write outputs.',
        type=str,
        default='./',
    )
    ag.add_argument(
        '-f',
        '--filename',
        help='Name of dataset, default is graph.pt',
        type=str,
        default='graph.pt',
    )
    ag.add_argument(
        '-ss',
        '--skip_statistics',
        help='Skip running & printing statistics',
        action='store_true',
    )
    ag.add_argument(
        '--legacy',
        help=legacy_help,
        action='store_true',
    )
    ag.add_argument(
        '--kwargs',
        nargs=argparse.REMAINDER,
        help='will be passed to ase.io.read, or can be used to specify EFS key',
    )

    args = ag.parse_args()
    return args
