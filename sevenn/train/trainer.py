from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sevenn._keys as KEY
from sevenn._const import LossType
from sevenn.error_recorder import ErrorRecorder
from sevenn.train.optim import loss_dict, optim_dict, scheduler_dict
from sevenn.util import AverageNumber, postprocess_output, squared_error


class Trainer:
    def __init__(self, model, config: dict):
        self.distributed = config[KEY.IS_DDP]

        if self.distributed:
            device = torch.device('cuda', config[KEY.LOCAL_RANK])
            dist.barrier()
            self.model = DDP(model.to(device), device_ids=[device])
            self.model.module.set_is_batch_data(True)
            self.rank = config[KEY.LOCAL_RANK]
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
        self.is_stress = self.is_trace_stress or self.is_train_stress
        if self.is_train_stress:
            self.loss_weights[LossType.STRESS] = config[KEY.STRESS_WEIGHT]

        # init loss_types
        self.loss_types = [LossType.ENERGY, LossType.FORCE]
        if self.is_stress:
            self.loss_types.append(LossType.STRESS)

        param = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim_dict[config[KEY.OPTIMIZER].lower()]
        optim_param = config[KEY.OPTIM_PARAM]
        self.optimizer = optimizer(param, **optim_param)

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

        self.fisher_information = config[KEY.CONTINUE][KEY.FISHER]
        self.optimal_params = config[KEY.CONTINUE][KEY.OPT_PARAMS]
        self._lambda = config[KEY.CONTINUE][KEY.EWC_LAMBDA]

        if self.fisher_information is not False and self.optimal_params is not False:
            self.set_ewc_settings(self.fisher_information, self.optimal_params)

        self.log = open('EWC_debug.log', 'w', buffering=1)

    def set_ewc_settings(self, fisher_information, optimal_params):
        import pickle

        with open(self.fisher_information, 'rb') as fr:
            fisher_dict = pickle.load(fr)
        with open(self.optimal_params, 'rb') as fr:
            opt_params_dict = pickle.load(fr)

        self.fisher_dict = fisher_dict
        self.opt_params_dict = opt_params_dict

    def run_one_epoch(
        self, loader, is_train=False, error_recorder: ErrorRecorder = None
    ):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        for step, batch in enumerate(loader):
            if is_train:
                self.optimizer.zero_grad()
            batch = batch.to(self.device, non_blocking=True)
            output = self.model(batch)
            error_recorder.update(output)
            if is_train:
                total_loss, ewc_loss = self.loss_calculator(output)
                total_loss.backward()
                self.optimizer.step()

        self.log.write(f'EWC loss: {ewc_loss}\n')

        if self.distributed:
            self.recorder_all_reduce(error_recorder)

    def loss_calculator_fisher(self, output):
        unit_converted = postprocess_output(output, self.loss_types)
        total_loss = torch.tensor([0.0], device=self.device)
        for loss_type in self.loss_types:
            pred, ref, _ = unit_converted[loss_type]
            total_loss +=\
                self.criterion(pred, ref) * self.loss_weights[loss_type]
        return total_loss

    def loss_calculator(self, output):
        unit_converted = postprocess_output(output, self.loss_types)
        total_loss = torch.tensor([0.0], device=self.device)
        for loss_type in self.loss_types:
            pred, ref, _ = unit_converted[loss_type]
            total_loss += (
                self.criterion(pred, ref) * self.loss_weights[loss_type]
            )

        ewc_loss = 0
        if self.fisher_information is not False and self.optimal_params is not False:
            for name, _param in self.model.named_parameters():
                if name in self.fisher_dict:
                    fisher = self.fisher_dict[name].to(self.device)
                    opt_param = self.opt_params_dict[name].to(self.device)
                    total_loss += (self._lambda / 2) * torch.sum(fisher * (_param - opt_param) ** 2)
                    ewc_loss += (self._lambda / 2) * torch.sum(fisher * (_param - opt_param) ** 2)

        return total_loss, ewc_loss

    def compute_fisher_matrix(self, loader):
        fisher_information = {}
        for name, _param in self.model.named_parameters():
            fisher_information[name] = torch.zeros_like(_param)

        self.model.train()
        for i, batch in enumerate(loader):
            self.model.zero_grad()
            batch = batch.to(self.device, non_blocking=True)
            output = self.model(batch)
            loss = self.loss_calculator_fisher(output)
            loss.backward()

            # Accumulating the squared gradients (Fisher information)
            for name, _param in self.model.named_parameters():
                if _param.grad is not None:
                    fisher_information[name] += _param.grad.data.clone() ** 2 

        dataset_size = len(loader.dataset)
        for name in fisher_information:
            fisher_information[name] /= dataset_size

        import pickle
        with open('fisher_sevenn.pkl', 'wb') as f:
            pickle.dump(fisher_information, f)
        optimal_params = {name: _param.data.detach().clone() for name, _param in self.model.named_parameters()}
        with open('opt_params_sevenn.pkl', 'wb') as f:
            pickle.dump(optimal_params, f)

        return fisher_information

    def scheduler_step(self, metric=None):
        if self.scheduler is None:
            return
        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def recorder_all_reduce(self, recorder: ErrorRecorder):
        for metric in recorder.metrics:
            # metric.value._ddp_reduce(self.device)
            metric.ddp_reduce(self.device)

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
        return {
            'model_state_dict': model_state_dct,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

    def load_state_dicts(
        self,
        model_state_dict,
        optimizer_state_dict,
        scheduler_state_dict,
        strict=True,
    ):
        if self.distributed:
            self.model.module.load_state_dict(model_state_dict, strict=strict)
        else:
            self.model.load_state_dict(model_state_dict, strict=strict)

        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        if scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)
