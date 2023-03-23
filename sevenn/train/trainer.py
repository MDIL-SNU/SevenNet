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
        total_atom_type = config[KEY.CHEMICAL_SPECIES]
        if 'total' not in user_labels:
            user_labels.insert(0, 'total')  # prepand total
        self.user_labels = user_labels
        self.total_atom_type = total_atom_type
        self.device = device
        self.force_weight = config[KEY.FORCE_WEIGHT]

        self.is_trace_stress = config[KEY.IS_TRACE_STRESS]
        self.is_train_stress = config[KEY.IS_TRAIN_STRESS]
        self.is_stress = (self.is_trace_stress or self.is_train_stress)

        if self.is_train_stress:
            self.stress_weight = config[KEY.STRESS_WEIGHT]

        self.param = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim_dict[config[KEY.OPTIMIZER].lower()]
        optim_param = config[KEY.OPTIM_PARAM]

        self.optimizer = optimizer(self.param, **optim_param)

        scheduler = scheduler_dict[config[KEY.SCHEDULER].lower()]
        scheduler_param = config[KEY.SCHEDULER_PARAM]
        self.scheduler = scheduler(self.optimizer, **scheduler_param)

        self.criterion = torch.nn.MSELoss(reduction='none')
        self.loss_hist = {}
        self.force_loss_hist_by_atom_type = {}

        for data_set_key in DataSetType:
            self.loss_hist[data_set_key] = {}
            self.force_loss_hist_by_atom_type[data_set_key] = {}
            for label in self.user_labels:
                loss_type_dict = {'energy': [], 'force': [], 'stress': []} if self.is_stress else {'energy': [], 'force': []}
                self.loss_hist[data_set_key][label] = loss_type_dict
            
            for atom_type in total_atom_type:
                self.force_loss_hist_by_atom_type[data_set_key][atom_type] = []


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
    """

    def get_vector_component_and_loss(self, scaled_pred_V: torch.Tensor,
                                scaled_ref_V: torch.Tensor, vdim: int):

        scaled_pred_V_component = torch.reshape(scaled_pred_V, (-1,))
        scaled_ref_V_component = torch.reshape(scaled_ref_V, (-1,))

        loss = self.criterion(scaled_pred_V_component, scaled_ref_V_component)
        loss = torch.reshape(loss, (-1, vdim))
        loss = loss.sum(dim=1)

        return scaled_pred_V_component, scaled_ref_V_component, loss
    

    def run_one_epoch(self, loader, set_type: DataSetType):
        is_train = True if set_type == DataSetType.TRAIN else False
        self.model.set_is_batch_data(True)
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = None

        epoch_loss = {}
        force_loss_by_atom_type = {}

        for label in self.user_labels:
            loss_type_dict = {'energy': [], 'force': [], 'stress': []} if self.is_stress else {'energy': [], 'force': []}
            epoch_loss[label] = loss_type_dict
        
        for atom_type in self.total_atom_type:
            force_loss_by_atom_type[atom_type] = []
        
        pred_E_set = []
        ref_E_set = []
        pred_F_set = []
        ref_F_set = []
        pred_S_set = []
        ref_S_set = []
        label_set = []
        atom_set = []

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
            
            scaled_pred_E_per_atom = torch.squeeze(result[KEY.SCALED_PER_ATOM_ENERGY], -1)
            scaled_ref_E_per_atom = batch[KEY.REF_SCALED_PER_ATOM_ENERGY]

            scaled_pred_F = result[KEY.SCALED_FORCE]
            scaled_ref_F = batch[KEY.REF_SCALED_FORCE]

            scaled_pred_F_component, scaled_ref_F_component, scaled_loss_F = \
                self.get_vector_component_and_loss(scaled_pred_F, scaled_ref_F, 3)

            scaled_loss_E = \
                self.criterion(scaled_pred_E_per_atom, scaled_ref_E_per_atom)

            scaled_loss_S = None

            if self.is_stress:
                scaled_pred_S = result[KEY.SCALED_STRESS] * 1602.1766208  # Convert to kB
                scaled_ref_S = batch[KEY.REF_SCALED_STRESS] * 1602.1766208  # Convert to kB
                scaled_pred_S_component, scaled_ref_S_component, scaled_loss_S = \
                    self.get_vector_component_and_loss(scaled_pred_S, scaled_ref_S, 6)
                

            label = batch[KEY.USER_LABEL]
            temp_atom_type_list = batch[KEY.CHEMICAL_SYMBOL]
            atom_type_list = []
            for chemical_symbols in temp_atom_type_list:
                atom_type_list.extend(chemical_symbols)

            epoch_loss = Trainer._update_epoch_loss(epoch_loss,
                                                    label,
                                                    batch[KEY.BATCH],
                                                    scaled_loss_E,
                                                    scaled_loss_F,
                                                    scaled_loss_S)
            force_loss_by_atom_type = Trainer._update_force_loss_by_atom_type(force_loss_by_atom_type, atom_type_list, scaled_loss_F)

            if is_train:
                loss1 = torch.mean(scaled_loss_E)
                loss2 = torch.mean(scaled_loss_F)
                total_loss = loss1 + self.force_weight * loss2 / 3
                if self.is_train_stress:
                    loss3 = torch.mean(scaled_loss_S)
                    total_loss = total_loss + self.stress_weight * loss3 / 6
                #clip_grad_norm(self.param, 5)
                total_loss.backward()
                self.optimizer.step()

            pred_F_set.extend(scaled_pred_F_component.tolist())
            ref_F_set.extend(scaled_ref_F_component.tolist())
            pred_E_set.extend(scaled_pred_E_per_atom.tolist())
            ref_E_set.extend(scaled_ref_E_per_atom.tolist())
            label_set.extend(label)
            atom_set.extend(atom_type_list)

            if self.is_stress:
                pred_S_set.extend(scaled_pred_S_component.tolist())
                ref_S_set.extend(scaled_ref_S_component.tolist())

        if is_train:
            self.scheduler.step()

        # self._update_loss_hist(set_type + '_loss')
        for key in self.user_labels:
            E_mse = np.mean(epoch_loss[key]['energy'])
            F_mse = np.mean(epoch_loss[key]['force'])
            self.loss_hist[set_type][key]['energy'].append(E_mse)
            self.loss_hist[set_type][key]['force'].append(F_mse)

            if self.is_stress:
                S_mse = np.mean(epoch_loss[key]['stress'])
                self.loss_hist[set_type][key]['stress'].append(S_mse)
            if key == 'total':
                # TODO: check
                loss_value = E_mse + self.force_weight * F_mse / 3
                if self.is_stress:
                    loss_value += self.stress_weight * S_mse / 6

        for key in self.total_atom_type:
            F_mse = np.mean(force_loss_by_atom_type[key])

            self.force_loss_hist_by_atom_type[set_type][key].append(F_mse)

        return pred_E_set, ref_E_set, pred_F_set, ref_F_set, pred_S_set, ref_S_set, label_set, atom_set, loss_value

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
    def _update_epoch_loss(epoch_loss, user_label: list, batch, E_loss: torch.Tensor,
                           F_loss: torch.Tensor, S_loss) -> dict:
        force_user_label = [user_label[int(i)] for i in batch]
        for key in epoch_loss.keys():
            if key == 'total':
                epoch_loss[key]['energy'].extend(E_loss.tolist())
                epoch_loss[key]['force'].extend(F_loss.tolist())
                if S_loss is not None:
                    epoch_loss[key]['stress'].extend(S_loss.tolist())
            else:
                energy_indices = [i for i, v in enumerate(user_label) if v == key]
                force_indices = [i for i, v in enumerate(force_user_label) if v == key]
                E_loss_list = E_loss[energy_indices]
                epoch_loss[key]['energy'].extend(E_loss_list.tolist())
                F_loss_list = F_loss[force_indices]
                epoch_loss[key]['force'].extend(F_loss_list.tolist())
                if S_loss is not None:
                    S_loss_list = S_loss[energy_indices]
                    epoch_loss[key]['stress'].extend(S_loss_list.tolist())

        return epoch_loss

    @staticmethod
    def _update_force_loss_by_atom_type(force_loss_by_atom_type, chemical_symbol: list, F_loss: torch.Tensor) -> dict:
        for key in force_loss_by_atom_type.keys():
            indices = [i for i, v in enumerate(chemical_symbol) if v == key]
            F_loss_list = F_loss[indices]
            force_loss_by_atom_type[key].extend(F_loss_list.tolist())

        return force_loss_by_atom_type

    """
    def _update_loss_hist(self, target_set: str):
        for key in self.loss_hist[target_set].keys():
            E_mse = np.mean(epoch_loss[key]['energy'])
            F_mse = np.mean(epoch_loss[key]['force'])
            self.loss_hist[target_set][key]['energy'].append(E_mse)
            self.loss_hist[target_set][key]['force'].append(F_mse)
    """
