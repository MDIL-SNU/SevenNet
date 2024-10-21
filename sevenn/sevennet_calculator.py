import os
import pathlib
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.jit
import torch.jit._script
from ase.calculators.calculator import Calculator, all_changes
from ase.data import chemical_symbols

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.train.dataload import unlabeled_atoms_to_graph

torch_script_type = torch.jit._script.RecursiveScriptModule


class SevenNetCalculator(Calculator):
    """ASE calculator for SevenNet models

    Multi-GPU parallel MD is not supported for this mode.
    Use LAMMPS for multi-GPU parallel MD.
    This class is for convenience who want to run SevenNet models with ase.

    Note than ASE calculator is designed to be interface of other programs.
    But in this class, we simply run torch model inside ASE calculator.
    So there is no FileIO things.

    Here, free_energy = energy
    """

    def __init__(
        self,
        model: Union[str, pathlib.PurePath, AtomGraphSequential] = '7net-0',
        file_type: str = 'checkpoint',
        device: Union[torch.device, str] = 'auto',
        sevennet_config: Optional[Any] = None,  # hold meta information
        **kwargs,
    ):
        """Initialize the calculator

        Args:
            model (SevenNet): path to the checkpoint file, or pretrained
            device (str, optional): Torch device to use. Defaults to "auto".
        """
        super().__init__(**kwargs)
        self.sevennet_config = None

        if isinstance(model, pathlib.PurePath):
            model = str(model)

        file_type = file_type.lower()
        if file_type not in ['checkpoint', 'torchscript', 'model_instance']:
            raise ValueError('file_type should be checkpoint or torchscript')

        if isinstance(device, str):  # TODO: do we really need this?
            if device == 'auto':
                self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        if file_type == 'checkpoint' and isinstance(model, str):
            if os.path.isfile(model):
                checkpoint = model
            else:
                checkpoint = util.pretrained_name_to_path(model)
            model_loaded, config = util.model_from_checkpoint(checkpoint)
            model_loaded.set_is_batch_data(False)
            self.type_map = config[KEY.TYPE_MAP]
            self.cutoff = config[KEY.CUTOFF]
            self.sevennet_config = config
        elif file_type == 'torchscript' and isinstance(model, str):
            extra_dict = {
                'chemical_symbols_to_index': b'',
                'cutoff': b'',
                'num_species': b'',
                'model_type': b'',
                'version': b'',
                'dtype': b'',
                'time': b'',
            }
            model_loaded = torch.jit.load(
                model, _extra_files=extra_dict, map_location=self.device
            )
            chem_symbols = extra_dict['chemical_symbols_to_index'].decode('utf-8')
            sym_to_num = {sym: n for n, sym in enumerate(chemical_symbols)}
            self.type_map = {
                sym_to_num[sym]: i for i, sym in enumerate(chem_symbols.split())
            }
            self.cutoff = float(extra_dict['cutoff'].decode('utf-8'))
        elif isinstance(model, AtomGraphSequential):
            if model.type_map is None:
                raise ValueError(
                    'Model must have the type_map to be used with calculator'
                )
            if model.cutoff == 0.0:
                raise ValueError('Model cutoff seems not initialized')
            model.eval_type_map = torch.tensor(True)  # ?
            model.set_is_batch_data(False)
            model_loaded = model
            self.type_map = model.type_map
            self.cutoff = model.cutoff
        else:
            raise ValueError('Unexpected input combinations')

        if self.sevennet_config is None and sevennet_config is not None:
            self.sevennet_config = sevennet_config

        self.model = model_loaded

        self.model.to(self.device)
        self.model.eval()

        self.implemented_properties = [
            'free_energy',
            'energy',
            'forces',
            'stress',
            'energies',
        ]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call parent class to set necessary atom attributes
        Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            raise ValueError('No atoms to evaluate')
        data = AtomGraphData.from_numpy_dict(
            unlabeled_atoms_to_graph(atoms, self.cutoff)
        )

        data.to(self.device)  # type: ignore

        if isinstance(self.model, torch_script_type):
            data[KEY.NODE_FEATURE] = torch.tensor(
                [self.type_map[z.item()] for z in data[KEY.NODE_FEATURE]],
                dtype=torch.int64,
                device=self.device,
            )
            data[KEY.POS].requires_grad_(True)  # backward compatibility
            data[KEY.EDGE_VEC].requires_grad_(True)  # backward compatibility
            data = data.to_dict()
            del data['data_info']

        output = self.model(data)
        energy = output[KEY.PRED_TOTAL_ENERGY].detach().cpu().item()
        # Store results
        self.results = {
            'free_energy': energy,
            'energy': energy,
            'energies': (
                output[KEY.ATOMIC_ENERGY].detach().cpu().reshape(len(atoms)).numpy()
            ),
            'forces': output[KEY.PRED_FORCE].detach().cpu().numpy(),
            'stress': np.array(
                (-output[KEY.PRED_STRESS])
                .detach()
                .cpu()
                .numpy()[[0, 1, 2, 4, 5, 3]]  # as voigt notation
            ),
        }
