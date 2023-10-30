from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sevenn._const import LossType
import sevenn._keys as KEY
from sevenn.util import postprocess_output, squared_error, AverageNumber
from sevenn.train.optim import optim_dict, scheduler_dict, loss_dict
from sevenn.error_recorder import ErrorRecorder

class Trainer():

    def __init__(self, model, config: dict):
        self.distributed = config[KEY.IS_DDP]

        if self.distributed:
            device = torch.device('cuda', config[KEY.LOCAL_RANK])
            dist.barrier()
            self.model = DDP(model.to(device), device_ids=[device])
            self.model.module.set_is_batch_data(True)
        else:
            device = config[KEY.DEVICE]
            self.model = model.to(device)
            self.model.set_is_batch_data(True)
        self.device = device

        self.loss_weights = {LossType.ENERGY: 1.0}
        self.loss_weights[LossType.FORCE] = config[KEY.FORCE_WEIGHT]

        # where trace stress is used for?
        self.is_trace_stress = config[KEY.IS_TRACE_STRESS]
        self.is_train_stress = config[KEY.IS_TRAIN_STRESS]
        self.is_stress = (self.is_trace_stress or self.is_train_stress)
        if self.is_train_stress:
            self.loss_weights[LossType.STRESS] = config[KEY.STRESS_WEIGHT]

        # init loss_types
        self.loss_types = [LossType.ENERGY, LossType.FORCE]
        if self.is_stress:
            self.loss_types.append(LossType.STRESS)

        self.param = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim_dict[config[KEY.OPTIMIZER].lower()]
        optim_param = config[KEY.OPTIM_PARAM]
        self.optimizer = optimizer(self.param, **optim_param)

        scheduler = scheduler_dict[config[KEY.SCHEDULER].lower()]
        scheduler_param = config[KEY.SCHEDULER_PARAM]
        self.scheduler = scheduler(self.optimizer, **scheduler_param)

        loss = loss_dict[config[KEY.LOSS].lower()]
        # TODO: handle this kind of case in parse_input not here
        try:
            loss_param = config[KEY.LOSS_PARAM]
        except KeyError:
            loss_param = {}
        self.criterion = loss(**loss_param)


    def run_one_epoch(self, loader, is_train=False,
                      error_recorder: ErrorRecorder = None):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        for step, batch in enumerate(loader):
            batch = batch.to(self.device, non_blocking=True)
            output = self.model(batch)
            error_recorder.update(output)
            if is_train:
                total_loss = self.loss_calculator(output)
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        if self.distributed:
            self.recorder_all_reduce(error_recorder)

    def loss_calculator(self, output):
        unit_converted = postprocess_output(output, self.loss_types)
        total_loss = torch.tensor([0.0], device=self.device)
        for loss_type in self.loss_types:
            pred, ref, _ = unit_converted[loss_type]
            total_loss +=\
                self.criterion(pred, ref) * self.loss_weights[loss_type]
        return total_loss

    def scheduler_step(self, metric=None):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def recorder_all_reduce(self, recorder: ErrorRecorder):
        for metric in recorder.metrics:
            metric.value.ddp_reduce(self.device)

    def _recursive_all_reduce(self, dct):
        for k, v in dct.items():
            if isinstance(v, dict):
                self._recursive_all_reduce(v)
            else:
                dist.all_reduce(v, op=dist.ReduceOp.SUM)

    # Not used, ddp automatically averages gradients
    def average_gradient(self):
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

    def get_checkpoint_dict(self):
        if self.distributed:
            model_state_dct = self.model.module.state_dict()
        else:
            model_state_dct = self.model.state_dict()
        return {'model_state_dict': model_state_dct,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()}

    """
    def _update_epoch_mse(self, epoch_mse, user_label: list,
                          natoms, mse_dct, epoch_counter):
        for loss_type in self.loss_types:
            epoch_mse['total'][loss_type] += torch.sum(mse_dct[loss_type])
            epoch_counter['total'][loss_type] += len(mse_dct[loss_type])

        mse_force = mse_dct[LossType.FORCE]
        for idx, label in enumerate(user_label):
            mse_labeled = epoch_mse[label]
            mse_labeled[LossType.ENERGY] += mse_dct[LossType.ENERGY][idx]
            epoch_counter[label][LossType.ENERGY] += 1
            if LossType.STRESS in self.loss_types:
                mse_labeled[LossType.STRESS] += mse_dct[LossType.STRESS][idx]
                epoch_counter[label][LossType.STRESS] += 1
            natom = natoms[idx].item()
            force_mse = torch.sum(mse_force[:natom])
            mse_force = mse_force[natom:]
            epoch_counter[label][LossType.FORCE] += natom
            mse_labeled[LossType.FORCE] += force_mse

    # Deprecated
    def _update_force_mse_by_atom_type(self, force_mse_by_atom_type,
                                       atomic_numbers: list,
                                       F_loss: torch.Tensor) -> dict:
        # write force_mse_by_atom_type dict, element wise force loss
        for key in force_mse_by_atom_type.keys():
            indices = [i for i, v in enumerate(atomic_numbers) if v == key]
            F_mse_list = F_loss[indices]
            force_mse_by_atom_type[key].extend(F_mse_list.tolist())
    """
