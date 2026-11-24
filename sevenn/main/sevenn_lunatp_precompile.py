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
        help='also build the kernels of second derivatives (force training)',
        action='store_true',
    )
    ag.add_argument(
        '--max_workers',
        help='processes building kernels. Default is min(8, available CPUs)',
        type=int,
        default=None,
    )


def run(args):
    import time

    from sevenn.nn.lunatp_helper import precompile

    start = time.perf_counter()
    order = 2 if args.train else 1
    precompile(args.checkpoint, order=order, max_workers=args.max_workers)
    print(f'LunaTP kernels ready in {time.perf_counter() - start:.1f} s')


def main(args=None):
    ag = argparse.ArgumentParser(description=description)
    add_args(ag)
    run(ag.parse_args())


if __name__ == '__main__':
    main()
