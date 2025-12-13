.. SevenNet documentation master file, created by
   sphinx-quickstart on Thu Nov 13 16:59:34 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/MDIL-SNU/SevenNet

SevenNet
========

SevenNet (Scalable EquiVariance-Enabled Neural Network) is a graph neural network (GNN)-based interatomic potential package that supports parallel molecular dynamics simulations using `LAMMPS <https://lammps.org>`_. Its core model is based on `NequIP <https://github.com/mir-group/nequip>`_.


* Pretrained GNN interatomic potential and fine-tuning interface
* Support for the Python Atomic Simulation Environment (ASE) calculator
* GPU-parallelized molecular dynamics with LAMMPS
* CUDA-accelerated D3 (van der Waals) dispersion
* Multi-fidelity training for combining multiple databases with different calculation settings


Installation
============
Ensure that your environment uses **Python >= 3.10** and **PyTorch >= 2.0.0** (see https://pytorch.org/get-started/locally/)
Then install SevenNet via:

.. code-block:: bash

    pip install sevenn

For acceleration and LAMMPS integration, refer to the the installation guides below.


Contents
========

.. toctree::
   :maxdepth: 2

   user_guide/index
   cite


.. toctree::
   :maxdepth: 1
   :caption: Misc

   changelog
