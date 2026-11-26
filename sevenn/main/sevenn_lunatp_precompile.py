import argparse

description = (
    'build the LunaTP kernels of a checkpoint in parallel before the first '
    'calculation. Run it in the environment of the later calculations, since '
    'the kernel cache key includes the Python environment and LD_LIBRARY_PATH'
)


def add_parser(subparsers):
    ag = subparsers.add_parser('lunatp_precompile', help=description)
    add_args(ag)


def add_args(parser):
    ag = parser
    ag.add_argument('checkpoint', help='checkpoint or pretrained', type=str)
    ag.add_argument(
        '--train',
        help='build second-derivative kernels for force training',
        action='store_true',
    )
    ag.add_argument(
        '--max_workers',
        help='processes building kernels. Default is min(16, available CPUs)',
        type=int,
        default=None,
    )
    widths = ag.add_mutually_exclusive_group()
    widths.add_argument(
        '--species_widths', nargs='+', type=int, default=(),
        help='also build self-connection kernels for these species counts',
    )
    widths.add_argument(
        '--all_species_widths', action='store_true',
        help='also build self-connection kernels for every possible species count',
    )


def run(args):
    import time

    from sevenn.nn.lunatp_helper import precompile

    start = time.perf_counter()
    order = 2 if args.train else 1
    if args.train and (args.species_widths or args.all_species_widths):
        raise SystemExit('Species widths apply to inference precompilation')
    if not precompile(
        args.checkpoint, order=order, max_workers=args.max_workers,
        species_widths=args.species_widths,
        all_species_widths=args.all_species_widths,
    ):
        raise SystemExit('LunaTP kernels could not be precompiled')
    print(f'LunaTP kernels ready in {time.perf_counter() - start:.1f} s')


def main(args=None):
    ag = argparse.ArgumentParser(description=description)
    add_args(ag)
    run(ag.parse_args())


if __name__ == '__main__':
    main()
