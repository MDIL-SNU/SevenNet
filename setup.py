from setuptools import find_packages, setup


setup(
    name="sevenn",
    version="0.8.1",
    description="SEVENNet",
    author="Yutack Park, Jaesun Kim",
    python_requires=">=3.8",
    packages=find_packages(include=["sevenn", "sevenn*"]),
    package_data={'': ['logo_ascii']},
    install_requires=[
        #"torch>=1.11",
        "ase",
        #"torch-geometric",
        "braceexpand",
        "pyyaml",
        "e3nn",
    ],
    entry_points={
        "console_scripts": [
            "sevenn = sevenn.main.sevenn:main",
            "sevenn_get_parallel = sevenn.main.sevenn_get_parallel:main"
        ]
    }
)


