import os
from typing import List, Union

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

import sevenn.util
from sevenn.nn.sequential import AtomGraphSequential
import sevenn._keys as KEY


class SevenNetCalculator(Calculator):
    """ASE calculator for SevenNet models

    Multi-GPU parallel MD is not supported for this mode.
    Use LAMMPS for multi-GPU parallel MD.
    This class is for convenience who want to run SevenNet models with ase.

    Note than ASE calculator is designed to be interface of other programs.
    But in this class, we simply run torch model inside ASE calculator.
    So there is no FileIO things.
    """

    def __init__(
        self,
        model: Union[AtomGraphSequential, str] = "SevenNet-0",
        device: Union[torch.device, str] = "cuda",
        sevennet_config=None,
        **kwargs
    ):
        """Initialize the calculator

        Args:
            model (SevenNet): AtomGraphSequential or path to the checkpoint file.
            device (str, optional): Torch device to use. Defaults to "cuda".
        """
        super().__init__(**kwargs)

        if not isinstance(model, AtomGraphSequential) and not isinstance(model, str):
            raise ValueError("model must be an instance of AtomGraphSequential or str.")
        if isinstance(model, str):
            # TODO: Download it from internet
            if model == "SevenNet-0":  # special case loading pre-trained model
                checkpoint = os.getenv("SEVENNET_0_CP")
                if checkpoint is None:
                    raise ValueError("Please set env variable SEVENNET_0_CP as checkpoint path.")
            else:
                checkpoint = model
            model, config = sevenn.util.model_from_checkpoint(checkpoint)
        else:
            if model._use_type_map is False and model.type_map is None:
                raise ValueError("model must have a type_map")
            model.set_use_type_map(True)
        self.sevennet_config = sevennet_config  # metadata which can be None
        try:
            self.cutoff = model.cutoff
        except AttributeError:
            self.cutoff = self.sevennet_config[KEY.CUTOFF]

        self.model = model

        if not isinstance(device, torch.device) and not isinstance(device, str):
            raise ValueError("device must be an instance of torch.device or str.")
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()
        self.model.set_is_batch_data(False)

        self.implemented_properties = ['energy', 'forces', 'stress']

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call parent class to set necessary atom attributes
        Calculator.calculate(self, atoms, properties, system_changes)
        data = sevenn.util.unlabeled_atoms_to_input(atoms, self.cutoff)
        data = self.model.to_onehot_idx(data)
        data.to(self.device)

        output = self.model(data)
        # Store results
        self.results = {
            'energy': output[KEY.PRED_TOTAL_ENERGY].detach().cpu().item(),
            'forces': output[KEY.PRED_FORCE].detach().cpu().numpy(),
            'stress': np.array((-output[KEY.PRED_STRESS]).detach().cpu().numpy()[[0, 1, 2, 4, 5, 3]])
        }
