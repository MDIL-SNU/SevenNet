from setuptools import find_packages, setup


setup(
    name="sevenn",
    version=0.90,
    description="SEVENN proto",
    author="Yutack Park, Jaesun Kim",
    python_requires=">=3.8",
    packages=find_packages(include=["sevenn", "sevenn*"]),
    install_requires=[
        "torch>=1.11",
        "ase",
        "torch-geometric",
        "braceexpand",
    ],
    entry_points={
        "console_scripts": [
            "sevenn = sevenn.scripts.sevenn_script:main"
        ]
    }
)


