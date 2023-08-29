from enum import Enum
from typing import Dict, Union

import torch
from torch.nn.utils import clip_grad_norm
from torch import linalg as LA
from torch_scatter import scatter
import numpy as np

import sevenn._keys as KEY
from sevenn.train.optim import optim_dict, scheduler_dict


#TODO: Optimizer type/parameter selection, loss type selection,
# for only one [title] in structure_list: not implemented, early stopping?

TO_KB = 1602.1766208  # eV/A^3 to kbar


class LossType(Enum):
    ENERGY = 'energy'
    FORCE = 'force'
    STRESS = 'stress'


class DataSetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class Trainer():

    def __init__(
        self, model, user_labels: list, config: dict,
        # TODO: replace per atom energy to total energy (divide here)
        #       remove from _keys.py
        energy_key: str = KEY.PRED_PER_ATOM_ENERGY,
        ref_energy_key: str = KEY.PER_ATOM_ENERGY,
        force_key: str = KEY.PRED_FORCE,
        ref_force_key: str = KEY.FORCE,
        stress_key: str = KEY.PRED_STRESS,
        ref_stress_key: str = KEY.STRESS,
        optimizer_state_dict=None, scheduler_state_dict=None
    ):
        """
        note that energy key is 'per-atom'
        """
        self.energy_key = energy_key
        self.ref_energy_key = ref_energy_key
        self.force_key = force_key
        self.ref_force_key = ref_force_key
        self.stress_key = stress_key
        self.ref_stress_key = ref_stress_key

        device = config[KEY.DEVICE]
        self.model = model.to(device)
        total_atom_type = config[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER]
        if 'total' not in user_labels:
            user_labels.insert(0, 'total')  # prepand total
        self.user_labels = user_labels
        self.total_atom_type = total_atom_type
        self.device = device
        self.force_weight = config[KEY.FORCE_WEIGHT]

        # where trace stress is used for?
        self.is_trace_stress = config[KEY.IS_TRACE_STRESS]
        self.is_train_stress = config[KEY.IS_TRAIN_STRESS]
        self.is_stress = (self.is_trace_stress or self.is_train_stress)
        if self.is_train_stress:
            self.stress_weight = config[KEY.STRESS_WEIGHT]
        else:
            self.stress_weight = 0

        # init loss_types
        self.loss_types = [LossType.ENERGY, LossType.FORCE]
        if self.is_stress:
            self.loss_types.append(LossType.STRESS)

        self.param = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim_dict[config[KEY.OPTIMIZER].lower()]
        optim_param = config[KEY.OPTIM_PARAM]
        self.optimizer = optimizer(self.param, **optim_param)

        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

        scheduler = scheduler_dict[config[KEY.SCHEDULER].lower()]
        scheduler_param = config[KEY.SCHEDULER_PARAM]
        self.scheduler = scheduler(self.optimizer, **scheduler_param)

        if scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)

        self.criterion = torch.nn.MSELoss(reduction='none')

        # initialize loss history containers
        # loss_hist is 3-dim: [DataSetType][Label][LossType]
        # force_loss_hist is 2-dim: [DataSetType][Specie]
        self.loss_hist = {}
        self.force_loss_hist_by_atom_type = {}
        for data_set_key in DataSetType:
            self.loss_hist[data_set_key] = {}
            self.force_loss_hist_by_atom_type[data_set_key] = \
                {at: [] for at in total_atom_type}

            for label in self.user_labels:
                self.loss_hist[data_set_key][label] = \
                    {lt: [] for lt in self.loss_types}

    def loss_function(self, loss_dct: Dict[LossType, Union[float, torch.Tensor]]):
        """
        for mean of mse of pred, ref pair
        """
        energy_loss = loss_dct[LossType.ENERGY]
        force_loss = loss_dct[LossType.FORCE]
        try:
            stress_loss = loss_dct[LossType.STRESS]
        except KeyError:
            stress_loss = 0
        return energy_loss + \
            self.force_weight * force_loss / 3 + \
            self.stress_weight * stress_loss / 6

    def postprocess_output(self, output, loss_type: LossType):
        # return pred, ref, loss
        if loss_type is LossType.ENERGY:
            pred = torch.squeeze(output[self.energy_key], -1)
            ref = output[self.ref_energy_key]
            loss = self.criterion(pred, ref)
        elif loss_type is LossType.FORCE:
            pred_raw = output[self.force_key]
            ref_raw = output[self.ref_force_key]
            pred, ref, loss = \
                self.get_vector_component_and_loss(pred_raw, ref_raw, 3)
        elif loss_type is LossType.STRESS:
            # calculate stress loss based on kB unit (was eV/A^3)
            pred_raw = output[self.stress_key] * TO_KB
            ref_raw = output[self.ref_stress_key] * TO_KB
            pred, ref, loss = \
                self.get_vector_component_and_loss(pred_raw, ref_raw, 6)
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')

        return pred, ref, loss

    def get_vector_component_and_loss(self, pred_V: torch.Tensor,
                                      ref_V: torch.Tensor, vdim: int):

        pred_V_component = torch.reshape(pred_V, (-1,))
        ref_V_component = torch.reshape(ref_V, (-1,))

        loss = self.criterion(pred_V_component, ref_V_component)
        loss = torch.reshape(loss, (-1, vdim))
        loss = loss.sum(dim=1)

        return pred_V_component, ref_V_component, loss

    def run_one_epoch(self, loader, set_type: DataSetType):
        is_train = set_type == DataSetType.TRAIN
        self.model.set_is_batch_data(True)
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        # actual loss tensor for backprop
        total_loss = None
        # container for print
        epoch_loss = {label:
                      {lt: [] for lt in self.loss_types}
                      for label in self.user_labels
                      }
        force_loss_by_atom_type = {at: [] for at in self.total_atom_type}

        parity_set = {"labels": [], "species": []}
        # save raw string instead of LossType for user
        # this is really strange... better way? see plot.py parity plot
        for y_value in ("energy", "force", "stress"):
            parity_set.update({y_value: {"pred": [], "ref": []}})

        for step, batch in enumerate(loader):
            batch.to(self.device)
            label = batch[KEY.USER_LABEL]
            atom_type_list = batch[KEY.ATOMIC_NUMBERS]
            self.optimizer.zero_grad(set_to_none=True)

            result = self.model(batch)

            loss_dct = {}
            for loss_type in self.loss_types:
                str_key = str(loss_type.value)
                pred, ref, loss = self.postprocess_output(result, loss_type)
                parity_set[str_key]["pred"].extend(pred.tolist())
                parity_set[str_key]["ref"].extend(ref.tolist())
                loss_dct[loss_type] = loss

            parity_set["labels"].extend(label)
            parity_set["species"].extend(atom_type_list)

            # store postprocessed results to history
            self._update_epoch_loss(
                epoch_loss, label, batch[KEY.BATCH], loss_dct
            )

            # store postprocessed results to history
            self._update_force_loss_by_atom_type(
                force_loss_by_atom_type,
                atom_type_list, loss_dct[LossType.FORCE]
            )

            if is_train:
                # mean inside loss_function? I don't know dim of loss_X
                # TODO: maybe safe to mean inside loss_function, fix later
                loss_dct = {k: torch.mean(v) for k, v in loss_dct.items()}
                total_loss = self.loss_function(loss_dct)
                total_loss.backward()
                self.optimizer.step()

        if is_train:
            self.scheduler.step()

        # self._update_loss_hist(set_type + '_loss')
        loss_record = {}
        for label in self.user_labels:
            loss_record[label] = {}
            for loss_type in self.loss_types:
                mse = np.mean(epoch_loss[label][loss_type])
                loss_record[label][loss_type] = mse
                self.loss_hist[set_type][label][loss_type].append(mse)

        specie_wise_loss_record = {}
        for atom_type in self.total_atom_type:
            F_mse = np.mean(force_loss_by_atom_type[atom_type])
            self.force_loss_hist_by_atom_type[set_type][atom_type].append(F_mse)
            specie_wise_loss_record[atom_type] = F_mse

        return parity_set, loss_record, specie_wise_loss_record

    def get_checkpoint_dict(self):
        return {'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': self.loss_hist}

    def _update_epoch_loss(self, epoch_loss, user_label: list,
                           batch, loss_dct) -> dict:
        f_user_label = [user_label[int(i)] for i in batch]

        for key in epoch_loss.keys():
            loss_labeld = epoch_loss[key]
            for loss_type in self.loss_types:
                loss = loss_dct[loss_type]
                if key == 'total':
                    loss_labeld[loss_type].extend(loss.tolist())
                    continue
                if loss_type is LossType.FORCE:
                    indicies = [i for i, v in enumerate(f_user_label) if v == key]
                else:  # energy or stress
                    indicies = [i for i, v in enumerate(user_label) if v == key]
                loss_indexed = loss[indicies]
                loss_labeld[loss_type].extend(loss_indexed.tolist())

    def _update_force_loss_by_atom_type(self, force_loss_by_atom_type,
                                        atomic_numbers: list,
                                        F_loss: torch.Tensor) -> dict:
        for key in force_loss_by_atom_type.keys():
            indices = [i for i, v in enumerate(atomic_numbers) if v == key]
            F_loss_list = F_loss[indices]
            force_loss_by_atom_type[key].extend(F_loss_list.tolist())
