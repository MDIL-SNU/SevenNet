from setuptools import find_packages, setup

setup(
    name='sevenn',
    version='0.9.0',
    description='SEVENNet',
    author='Yutack Park, Jaesun Kim',
    python_requires='>=3.8',
    packages=find_packages(include=['sevenn', 'sevenn*']),
    package_data={'': ['logo_ascii']},
    install_requires=[
        # "torch>=1.11",
        'ase',
        # "torch-geometric",
        'braceexpand',
        'pyyaml',
        'e3nn',
    ],
    entry_points={
        'console_scripts': [
            'sevenn = sevenn.main.sevenn:main',
            'sevenn_get_model = sevenn.main.sevenn_get_model:main',
            'sevenn_graph_build = sevenn.main.sevenn_graph_build:main',
            'sevenn_inference = sevenn.main.sevenn_inference:main',
        ]
    },
)
