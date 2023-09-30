import os
import sys
import argparse
from datetime import datetime

import sevenn.scripts.graph_build as graph_build
from sevenn.sevenn_logger import Logger
from sevenn._const import SEVENN_VERSION
import sevenn._keys as KEY

description = f"sevenn version={SEVENN_VERSION}, sevenn_graph_build, "\
    + "build graph from ase readable or ase atoms or VASP OUTCARs (by structure_list)\n"\
    + "CAUTION: If the ase.atoms are not from ase read of VASP, you must check units "\
    + "energy: eV (vasp free energy), force: eV/Angstrom, stress: eV/Angstrom^3 " \
    + "and stress tensor to be 6x1 vector in order of xx, yy, zz, yz, xz, xy of INTERNAL stress "\
    + "What you see from VASP(grep 'in kB') is external stress in kB "

source_help = "primitive data to build graph, assume structure_list if format is not given "\
    + "or assume root dir of pickles of ase.Atoms list if directory"
label_by_help = "label the given dataset with given string, "\
    + "if the src is dir of .pkl, default is the name of the directory "\
    + "if the src is structure_list, default is the label of the structure_list file"
cutoff_help = "cutoff radius for graph building in Angstrom "
suffix_help = "when source is dir, suffix of files. if not given, read all recursively"\
    + "ignored if source is structure_list"
copy_info_help = "do not copy ase.Atoms.info to graph data, "\
    + "ignored if source is sturcture_list"
format_help = "format of the source, defualt is structure_list "\
    + "if it is pkl/pickle, assume they are list of ase.Atoms, "\
    + "else, the input is directly passed to ase.io.read"

def main(args=None):
    metadata = {}
    now = datetime.now().strftime('%Y-%m-%d')
    source, cutoff, num_cores, label_by, out, save_by_label, fmt, suffix, copy_info = \
        cmd_parse_data(args)
    metadata = {"sevenn_version": SEVENN_VERSION,
                "when": now, "cutoff": cutoff}
    Logger("graph_build_log", screen=True)
    Logger().writeline(description)
    if not os.path.exists(source):
        raise ValueError(f"source {source} does not exist")

    graph_build.build_script(source, cutoff, num_cores,
                             label_by, out, save_by_label, fmt,
                             suffix, copy_info, metadata)

def cmd_parse_data(args=None):
    ag = argparse.ArgumentParser(description=description)

    ag.add_argument('source',
                    help=source_help,
                    type=str)
    ag.add_argument('cutoff',
                    help=cutoff_help,
                    type=float)
    ag.add_argument('-n', '--num_cores',
                    help='number of cores to build graph in parallel',
                    default=1,
                    type=int)
    ag.add_argument('-l', '--label_by',
                    help=label_by_help,
                    default="auto",
                    type=str)
    ag.add_argument('-f', '--format',
                    help=format_help,
                    type=str,
                    default='structure_list')
    """
    ag.add_argument('-m', '--merge',
                    help='merge the whole dataset into one file',
                    action='store_true',
                    default=False)
    """
    ag.add_argument('-s', '--suffix',
                    help=suffix_help,
                    type=str,
                    default=None)
    ag.add_argument('-nc', '--no_copy_info',
                    help=copy_info_help,
                    action='store_true',
                    default=False)
    ag.add_argument('-o', '--out',
                    help='filename or dir path to write outputs',
                    type=str,
                    default="./")
    ag.add_argument('-sb', '--save_by_label',
                    help='save the graph by label',
                    action='store_true',
                    default=False)

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
    return source, cutoff, num_cores, label_by, out, save_by_label, fmt, suffix, copy_info
