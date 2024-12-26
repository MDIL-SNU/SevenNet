import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildLibPairD3(build_ext):
    def build_extension(self, ext):
        if not ext.name == 'libpaird3':
            super().build_extension(ext)
            return

        try:
            subprocess.run(
                ['nvcc', '--version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            print('CUDA is installed. Starting compilation of libpaird3.')
        except FileNotFoundError as e:
            print(
                'CUDA is not installed or nvcc is not available.'
                'Skipping compilation of libpaird3.'
            )
            return

        compile = [
            'nvcc',
            '-o', 'sevenn/libpaird3.so',
            '-shared',
            '-fmad=false',
            '-O3',
            '--expt-relaxed-constexpr',
            'sevenn/pair_e3gnn/pair_d3_for_ase.cu',
            '-Xcompiler', '-fPIC', '-lcudart',
            '-gencode', 'arch=compute_61,code=sm_61',
            '-gencode', 'arch=compute_70,code=sm_70',
            '-gencode', 'arch=compute_75,code=sm_75',
            '-gencode', 'arch=compute_80,code=sm_80',
            '-gencode', 'arch=compute_86,code=sm_86',
            '-gencode', 'arch=compute_89,code=sm_89',
            '-gencode', 'arch=compute_90,code=sm_90',
        ]  # you can add more architectures here

        try:
            subprocess.run(compile, check=True)
            print('libpaird3.so compiled successfully.')
        except subprocess.CalledProcessError as e:
            print(f'Failed to compile libpaird3.so: {e}')
            return


setup(
    ext_modules=[Extension('libpaird3', sources=[])],
    cmdclass={'build_ext': BuildLibPairD3},
)
