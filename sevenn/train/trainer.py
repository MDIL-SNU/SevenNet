from enum import Enum

import torch
from torch.nn.utils import clip_grad_norm
from torch import linalg as LA
from torch_scatter import scatter
import numpy as np

import sevenn._keys as KEY
from sevenn.train.optim import optim_dict, scheduler_dict


#TODO: Optimizer type/parameter selection, loss type selection,
# for only one [title] in structure_list: not implemented, early stopping?

# Delete?
"""
def divide_by_atoms(pred_E: torch.Tensor, ref_E: torch.Tensor, batch_size, device):
    ones = torch.ones(batch_size.size()[0]).to(device)
    # change to torch.bincount?
    n_atoms = scatter(ones, batch_size, dim=0).unsqueeze(-1)
    pred_E = torch.div(pred_E, n_atoms)
    ref_E = torch.div(ref_E, n_atoms)
    return pred_E, ref_E
"""


class DataSetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class Trainer():

    def __init__(self, model, user_labels: list, config: dict):
        device = config[KEY.DEVICE]
        self.model = model.to(device)
        if 'Total' not in user_labels:
            user_labels.insert(0, 'Total')  # prepand total
        self.user_labels = user_labels
        self.device = device
        self.force_weight = config[KEY.FORCE_WEIGHT]

        self.param = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim_dict[config[KEY.OPTIMIZER].lower()]
        optim_param = config[KEY.OPTIM_PARAM]

        self.optimizer = optimizer(self.param, **optim_param)

        scheduler = scheduler_dict[config[KEY.SCHEDULER].lower()]
        scheduler_param = config[KEY.SCHEDULER_PARAM]
        self.scheduler = scheduler(self.optimizer, **scheduler_param)

        self.criterion = torch.nn.MSELoss(reduction='none')
        self.loss_hist = {}

        for data_set_key in DataSetType:
            self.loss_hist[data_set_key] = {}
            for label in self.user_labels:
                self.loss_hist[data_set_key][label] = {'energy': [], 'force': []}

        """
        self.loss_hist = {'train_loss': {'total': {'energy': [], 'force': []}},
                          'valid_loss': {'total': {'energy': [], 'force': []}},
                          'test_loss': {'total': {'energy': [], 'force': []}}}
        """
        # self.epoch_loss = {'total': {'energy': [], 'force': []}}

        #initialize loss_hist for each graph types.
        """
        for label in self.user_label:
            for key in self.loss_hist.keys():
                self.loss_hist[key].update({label: {'energy': [], 'force': []}})
        """

    def get_norm_and_loss_force(self, scaled_pred_F: torch.Tensor,
                                scaled_ref_F: torch.Tensor):
        scaled_pred_F_norm = LA.norm(scaled_pred_F, dim=1).tolist()
        scaled_ref_F_norm = LA.norm(scaled_ref_F, dim=1).tolist()

        pred_temp = torch.reshape(scaled_pred_F, (-1,))
        ref_temp = torch.reshape(scaled_ref_F, (-1,))

        loss = self.criterion(pred_temp, ref_temp)
        loss = torch.reshape(loss, (-1, 3))
        loss = loss.sum(dim=1)

        return scaled_pred_F_norm, scaled_ref_F_norm, loss

    def run_one_epoch(self, loader, set_type: DataSetType):
        is_train = True if set_type == DataSetType.TRAIN else False
        self.model.set_is_batch_data(True)
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = None

        epoch_loss = {}
        for label in self.user_labels:
            epoch_loss[label] = {'energy': [], 'force': []}

        pred_E_set = []
        ref_E_set = []
        pred_F_set = []
        ref_F_set = []
        label_set = []

        for step, batch in enumerate(loader):
            batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            # no compil mode
            result = self.model(batch)

            # compile mode
            """
            label = batch[KEY.USER_LABEL]
            del batch[KEY.USER_LABEL]  # since this can't be tensor it cause error
            result = self.model(batch.to_dict())
            """

            scaled_pred_E_per_atom = result[KEY.SCALED_PER_ATOM_ENERGY].squeeze()
            scaled_ref_E_per_atom = batch[KEY.REF_SCALED_PER_ATOM_ENERGY]

            scaled_pred_F = result[KEY.SCALED_FORCE]
            scaled_ref_F = batch[KEY.REF_SCALED_FORCE]

            scaled_pred_F_norm, scaled_ref_F_norm, scaled_loss_F = \
                self.get_norm_and_loss_force(scaled_pred_F, scaled_ref_F)

            scaled_loss_E = \
                self.criterion(scaled_pred_E_per_atom, scaled_ref_E_per_atom)

            label = batch[KEY.USER_LABEL]
            epoch_loss = Trainer._update_epoch_loss(epoch_loss,
                                                    label,
                                                    scaled_loss_E,
                                                    scaled_loss_F)

            if is_train:
                loss1 = torch.mean(scaled_loss_E)
                loss2 = torch.mean(scaled_loss_F)
                total_loss = loss1 + self.force_weight * loss2 / 3
                #clip_grad_norm(self.param, 5)
                total_loss.backward()
                self.optimizer.step()

            pred_F_set.extend(scaled_pred_F_norm)
            ref_F_set.extend(scaled_ref_F_norm)
            pred_E_set.extend(scaled_pred_E_per_atom.tolist())
            ref_E_set.extend(scaled_ref_E_per_atom.tolist())
            label_set.extend(label)

        if is_train:
            self.scheduler.step()

        # self._update_loss_hist(set_type + '_loss')
        for key in self.user_labels:
            E_mse = np.mean(epoch_loss[key]['energy'])
            F_mse = np.mean(epoch_loss[key]['force'])
            if key == 'total':
                # TODO: check
                loss_value = E_mse + self.force_weight * F_mse / 3
            self.loss_hist[set_type][key]['energy'].append(E_mse)
            self.loss_hist[set_type][key]['force'].append(F_mse)

        return pred_E_set, ref_E_set, pred_F_set, ref_F_set, label_set, loss_value

    def get_checkpoint_dict(self):
        return {'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': self.loss_hist}

    @staticmethod
    def from_checkpoint_dict(checkpoint):
        # TODO: implement this
        config = checkpoint['config']
        pass

    @staticmethod
    def _update_epoch_loss(epoch_loss, user_label: list, E_loss: torch.Tensor,
                           F_loss: torch.Tensor) -> dict:
        for key in epoch_loss.keys():
            if key == 'total':
                epoch_loss[key]['energy'].extend(E_loss.tolist())
                epoch_loss[key]['force'].extend(F_loss.tolist())
            else:
                indicies = [i for i, v in enumerate(user_label) if v == key]
                E_loss_list = E_loss[indicies]
                epoch_loss[key]['energy'].extend(E_loss_list.tolist())
                F_loss_list = F_loss[indicies]
                epoch_loss[key]['force'].extend(F_loss_list.tolist())
        return epoch_loss

    """
    def _update_loss_hist(self, target_set: str):
        for key in self.loss_hist[target_set].keys():
            E_mse = np.mean(epoch_loss[key]['energy'])
            F_mse = np.mean(epoch_loss[key]['force'])
            self.loss_hist[target_set][key]['energy'].append(E_mse)
            self.loss_hist[target_set][key]['force'].append(F_mse)
    """
