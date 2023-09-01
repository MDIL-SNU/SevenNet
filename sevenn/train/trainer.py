from enum import Enum
from typing import Dict, Union

import torch
from torch.nn.utils import clip_grad_norm
from torch import linalg as LA
from torch_scatter import scatter
import numpy as np

import sevenn._keys as KEY
from sevenn.train.optim import optim_dict, scheduler_dict, loss_dict


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
        optimizer_state_dict=None, scheduler_state_dict=None,
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

        loss = loss_dict[config[KEY.LOSS].lower()] # class
        # TODO: handle this kind of case in parse_input not here
        try:
            loss_param = config[KEY.LOSS_PARAM]
        except KeyError:
            loss_param = {}
        # TODO: is reduction param universal in pytorch? should I set it to none?
        self.criterion = loss(**loss_param)

        self.mse = torch.nn.MSELoss(reduction='none')
        #self.criterion = torch.nn.MSELoss(reduction='none')

        # initialize loss history containers
        # mse_hist is 3-dim: [DataSetType][Label][LossType]
        # force_mse_hist is 2-dim: [DataSetType][Specie]

        self.mse_hist = {}
        self.force_mse_hist_by_atom_type = {}
        for data_set_key in DataSetType:
            self.mse_hist[data_set_key] = {}
            self.force_mse_hist_by_atom_type[data_set_key] = \
                {at: [] for at in total_atom_type}
            for label in self.user_labels:
                self.mse_hist[data_set_key][label] = \
                    {lt: [] for lt in self.loss_types}

    """
    def mse_loss(self, mse_dct):
        energy_loss = mse_dct[LossType.ENERGY]
        force_loss = mse_dct[LossType.FORCE]
        try:
            stress_loss = mse_dct[LossType.STRESS]
        except KeyError:
            stress_loss = 0
        return energy_loss + \
            self.force_weight * force_loss / 3 + \
            self.stress_weight * stress_loss / 6
    """

    def postprocess_output(self, output, loss_type: LossType, is_train: bool):
        # from the output of model, calculate mse, loss
        # since they're all LossType wise
        def get_vector_component_and_mse(pred_V: torch.Tensor,
                                          ref_V: torch.Tensor, vdim: int):
            pred_V_component = torch.reshape(pred_V, (-1,))
            ref_V_component = torch.reshape(ref_V, (-1,))
            mse = self.mse(pred_V_component, ref_V_component)
            mse = torch.reshape(mse, (-1, vdim))
            mse = mse.sum(dim=1)
            return pred_V_component, ref_V_component, mse

        loss_weight = 0
        if loss_type is LossType.ENERGY:
            pred = torch.squeeze(output[self.energy_key], -1)
            ref = output[self.ref_energy_key]
            mse = self.mse(pred, ref)
            loss_weight = 1 # energy loss weight is 1 (it is reference)
        elif loss_type is LossType.FORCE:
            pred_raw = output[self.force_key]
            ref_raw = output[self.ref_force_key]
            pred, ref, mse = \
                get_vector_component_and_mse(pred_raw, ref_raw, 3)
            loss_weight = self.force_weight / 3  # normalize by # of comp
        elif loss_type is LossType.STRESS:
            # calculate stress loss based on kB unit (was eV/A^3)
            pred_raw = output[self.stress_key] * TO_KB
            ref_raw = output[self.ref_stress_key] * TO_KB
            pred, ref, mse = \
                get_vector_component_and_mse(pred_raw, ref_raw, 6)
            loss_weight = self.stress_weight / 6 # normalize by # of comp
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')

        loss = None
        if is_train:
            if isinstance(self.criterion, torch.nn.MSELoss):
                # mse for L2 loss is already calculated
                loss = torch.mean(mse) * loss_weight
            else:
                loss = self.criterion(pred, ref) * loss_weight

        return pred, ref, mse, loss

    def run_one_epoch(self, loader, set_type: DataSetType):
        # set model to train/eval mode
        is_train = set_type == DataSetType.TRAIN
        self.model.set_is_batch_data(True)
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        # initialize dict variables
        # actual loss tensor for backprop
        # container for print
        epoch_mse = {label:
                      {lt: [] for lt in self.loss_types}
                      for label in self.user_labels}
        force_mse_by_atom_type = {at: [] for at in self.total_atom_type}

        # iterate over batch
        for step, batch in enumerate(loader):
            batch.to(self.device)
            label = batch[KEY.USER_LABEL]
            atom_type_list = batch[KEY.ATOMIC_NUMBERS]
            self.optimizer.zero_grad(set_to_none=True)

            # forward
            result = self.model(batch)

            mse_dct = {}
            loss_dct = {}
            total_loss = None
            for loss_type in self.loss_types:
                # loss is ignored if it is_train false
                pred, ref, mse, loss = \
                    self.postprocess_output(result, loss_type, is_train)
                total_loss = loss if total_loss is None else total_loss + loss
                mse_dct[loss_type] = mse
                if is_train:
                    loss_dct[loss_type] = loss.item()

            # store postprocessed results to history
            # Before mean loss is required for structure-wise loss print
            self._update_epoch_mse(
                epoch_mse, label, batch[KEY.BATCH], mse_dct
            )
            self._update_force_mse_by_atom_type(
                force_mse_by_atom_type,
                atom_type_list, mse_dct[LossType.FORCE]
            )
            if is_train:
                total_loss.backward()
                self.optimizer.step()
        if is_train:
            self.scheduler.step()

        mse_record = {}
        for label in self.user_labels:
            mse_record[label] = {}
            for loss_type in self.loss_types:
                mse = np.mean(epoch_mse[label][loss_type])
                mse_record[label][loss_type] = mse
                self.mse_hist[set_type][label][loss_type].append(mse)

        specie_wise_mse_record = {}
        for atom_type in self.total_atom_type:
            F_mse = np.mean(force_mse_by_atom_type[atom_type])
            self.force_mse_hist_by_atom_type[set_type][atom_type].append(F_mse)
            specie_wise_mse_record[atom_type] = F_mse

        return mse_record, specie_wise_mse_record, loss_dct

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

    def get_checkpoint_dict(self):
        return {'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': self.mse_hist}  # not loss, mse

    def _update_epoch_mse(self, epoch_mse, user_label: list,
                           batch, mse_dct) -> dict:
        f_user_label = [user_label[int(i)] for i in batch]

        for key in epoch_mse.keys():
            mse_labeld = epoch_mse[key]
            for loss_type in self.loss_types:
                mse = mse_dct[loss_type]
                if key == 'total':
                    mse_labeld[loss_type].extend(mse.tolist())
                    continue
                if loss_type is LossType.FORCE:
                    indicies = [i for i, v in enumerate(f_user_label) if v == key]
                else:  # energy or stress
                    indicies = [i for i, v in enumerate(user_label) if v == key]
                mse_indexed = mse[indicies]
                mse_labeld[loss_type].extend(mse_indexed.tolist())

    def _update_force_mse_by_atom_type(self, force_mse_by_atom_type,
                                        atomic_numbers: list,
                                        F_loss: torch.Tensor) -> dict:
        for key in force_mse_by_atom_type.keys():
            indices = [i for i, v in enumerate(atomic_numbers) if v == key]
            F_mse_list = F_loss[indices]
            force_mse_by_atom_type[key].extend(F_mse_list.tolist())
