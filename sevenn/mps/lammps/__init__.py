"""LAMMPS on the Apple GPU: a serial pair style e3gnn that runs the model in Python
(src/, pair.py), and a `fix external` bridge for runs driven from Python
(external.py). TorchScript cannot hold the Metal kernels, so there is no deploy step:
the pair style loads the checkpoint itself."""
