from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

import sevenn._keys as KEY

from .atom_graph_data import AtomGraphData
from .train.optim import loss_dict

_ERROR_TYPES = {
    'TotalEnergy': {
        'name': 'Energy',
        'ref_key': KEY.ENERGY,
        'pred_key': KEY.PRED_TOTAL_ENERGY,
        'unit': 'eV',
        'vdim': 1,
    },
    'Energy': {  # by default per-atom for energy
        'name': 'Energy',
        'ref_key': KEY.ENERGY,
        'pred_key': KEY.PRED_TOTAL_ENERGY,
        'unit': 'eV/atom',
        'per_atom': True,
        'vdim': 1,
    },
    'Force': {
        'name': 'Force',
        'ref_key': KEY.FORCE,
        'pred_key': KEY.PRED_FORCE,
        'unit': 'eV/Å',
        'vdim': 3,
    },
    'Stress': {
        'name': 'Stress',
        'ref_key': KEY.STRESS,
        'pred_key': KEY.PRED_STRESS,
        'unit': 'kbar',
        'coeff': 1602.1766208,
        'vdim': 6,
    },
    'Stress_GPa': {
        'name': 'Stress',
        'ref_key': KEY.STRESS,
        'pred_key': KEY.PRED_STRESS,
        'unit': 'GPa',
        'coeff': 160.21766208,
        'vdim': 6,
    },
    'TotalLoss': {
        'name': 'TotalLoss',
        'unit': None,
    },
}


def get_err_type(name: str) -> dict[str, Any]:
    return deepcopy(_ERROR_TYPES[name])


class AverageNumber:
    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, values: torch.Tensor):
        self._sum += values.sum().item()
        self._count += values.numel()

    def _ddp_reduce(self, device):
        _sum = torch.tensor(self._sum, device=device)
        _count = torch.tensor(self._count, device=device)
        dist.all_reduce(_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(_count, op=dist.ReduceOp.SUM)
        self._sum = _sum.item()
        self._count = _count.item()

    def get(self):
        if self._count == 0:
            return torch.nan
        return self._sum / self._count


class ErrorMetric:
    """
    Base class for error metrics We always average error by # of structures,
    and designed to collect errors in the middle of iteration (by AverageNumber)
    """

    def __init__(
        self,
        name: str,
        ref_key: str,
        pred_key: str,
        coeff: float = 1.0,
        unit: Optional[str] = None,
        per_atom: bool = False,
        **kwargs,
    ):
        self.name = name
        self.unit = unit
        self.coeff = coeff
        self.ref_key = ref_key
        self.pred_key = pred_key
        self.per_atom = per_atom
        self.value = AverageNumber()

    def update(self, output: AtomGraphData):
        raise NotImplementedError

    def _retrieve(self, output: AtomGraphData):
        y_ref = output[self.ref_key] * self.coeff
        y_pred = output[self.pred_key] * self.coeff
        if self.per_atom:
            assert y_ref.dim() == 1 and y_pred.dim() == 1
            natoms = output[KEY.NUM_ATOMS]
            y_ref = y_ref / natoms
            y_pred = y_pred / natoms
        return y_ref, y_pred

    def ddp_reduce(self, device):
        self.value._ddp_reduce(device)

    def reset(self):
        self.value = AverageNumber()

    def get(self):
        return self.value.get()

    def key_str(self, with_unit=True):
        if self.unit is None or not with_unit:
            return self.name
        else:
            return f'{self.name} ({self.unit})'

    def __str__(self):
        return f'{self.key_str()}: {self.value.get():.6f}'


class RMSError(ErrorMetric):
    """
    Vector squared error
    """

    def __init__(self, vdim: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.vdim = vdim
        self._se = torch.nn.MSELoss(reduction='none')

    def _square_error(self, y_ref, y_pred, vdim: int):
        return self._se(y_ref.view(-1, vdim), y_pred.view(-1, vdim)).sum(dim=1)

    def update(self, output: AtomGraphData):
        y_ref, y_pred = self._retrieve(output)
        se = self._square_error(y_ref, y_pred, self.vdim)
        self.value.update(se)

    def get(self):
        return self.value.get() ** 0.5


class ComponentRMSError(ErrorMetric):
    """
    Ignore vector dim and just average over components
    Results smaller error
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._se = torch.nn.MSELoss(reduction='none')

    def _square_error(self, y_ref, y_pred):
        return self._se(y_ref, y_pred)

    def update(self, output: AtomGraphData):
        y_ref, y_pred = self._retrieve(output)
        y_ref = y_ref.view(-1)
        y_pred = y_pred.view(-1)
        se = self._square_error(y_ref, y_pred)
        self.value.update(se)

    def get(self):
        return self.value.get() ** 0.5


class MAError(ErrorMetric):
    """
    Average over all component
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _square_error(self, y_ref, y_pred):
        return torch.abs(y_ref - y_pred)

    def update(self, output: AtomGraphData):
        y_ref, y_pred = self._retrieve(output)
        y_ref = y_ref.reshape((-1,))
        y_pred = y_pred.reshape((-1,))
        se = self._square_error(y_ref, y_pred)
        self.value.update(se)


class CustomError(ErrorMetric):
    """
    Custom error metric
    Args:
        func: a function that takes y_ref and y_pred
              and returns a list of errors
    """

    def __init__(self, func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def update(self, output: AtomGraphData):
        y_ref, y_pred = self._retrieve(output)
        se = self.func(y_ref, y_pred)
        self.value.update(se)


class CombinedError(ErrorMetric):
    """
    Combine multiple error metrics with weights
    corresponds to a weighted sum of errors (normally used in loss)
    """

    def __init__(self, metrics: List[Tuple[ErrorMetric, float]], **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
        assert kwargs['unit'] is None

    def update(self, output: AtomGraphData):
        for metric, _ in self.metrics:
            metric.update(output)

    def reset(self):
        for metric, _ in self.metrics:
            metric.reset()

    def ddp_reduce(self, device):  # override
        for metric, _ in self.metrics:
            metric.value._ddp_reduce(device)

    def get(self):
        val = 0.0
        for metric, weight in self.metrics:
            val += metric.get() * weight
        return val


class ErrorRecorder:
    """
    record errors of a model
    """

    METRIC_DICT = {
        'RMSE': RMSError,
        'ComponentRMSE': ComponentRMSError,
        'MAE': MAError,
        'Loss': CustomError,
    }

    def __init__(self, metrics: List[ErrorMetric]):
        self.history = []
        self.metrics = metrics

    def _update(self, output: AtomGraphData):
        for metric in self.metrics:
            metric.update(output)

    def update(self, output: AtomGraphData, no_grad=True):
        if no_grad:
            with torch.no_grad():
                self._update(output)
        else:
            self._update(output)

    def get_metric_dict(self, with_unit=True):
        return {metric.key_str(with_unit): metric.get() for metric in self.metrics}

    def get_current(self):
        dct = {}
        for metric in self.metrics:
            dct[metric.name] = {
                'value': metric.get(),
                'unit': metric.unit,
                'ref_key': metric.ref_key,
                'pred_key': metric.pred_key,
            }
        return dct

    def get_dct(self, prefix=''):
        dct = {}
        if prefix.endswith('_') is False and prefix != '':
            prefix = prefix + '_'
        for metric in self.metrics:
            dct[f'{prefix}{metric.name}'] = f'{metric.get():6f}'
        return dct

    def get_key_str(self, name: str):
        for metric in self.metrics:
            if name == metric.name:
                return metric.key_str()
        return None

    def epoch_forward(self):
        self.history.append(self.get_current())
        pretty = self.get_metric_dict(with_unit=True)
        for metric in self.metrics:
            metric.reset()
        return pretty  # for print

    @staticmethod
    def init_total_loss_metric(config, criteria):
        is_stress = config[KEY.IS_TRAIN_STRESS]
        metrics = []
        energy_metric = CustomError(criteria, **get_err_type('Energy'))
        metrics.append((energy_metric, 1))
        force_metric = CustomError(criteria, **get_err_type('Force'))
        metrics.append((force_metric, config[KEY.FORCE_WEIGHT]))
        if is_stress:
            stress_metric = CustomError(criteria, **get_err_type('Stress'))
            metrics.append((stress_metric, config[KEY.STRESS_WEIGHT]))
        total_loss_metric = CombinedError(
            metrics, name='TotalLoss', unit=None, ref_key=None, pred_key=None
        )
        return total_loss_metric

    @staticmethod
    def from_config(config: dict):
        loss_cls = loss_dict[config.get(KEY.LOSS, 'mse').lower()]
        loss_param = config.get(KEY.LOSS_PARAM, {})
        criteria = loss_cls(**loss_param)

        err_config = config.get(KEY.ERROR_RECORD, False)
        if not err_config:
            raise ValueError(
                'No error_record config found. Consider util.get_error_recorder'
            )
        err_config_n = []
        if not config.get(KEY.IS_TRAIN_STRESS, True):
            for err_type, metric_name in err_config:
                if 'Stress' in err_type:
                    continue
                err_config_n.append((err_type, metric_name))
            err_config = err_config_n

        err_metrics = []
        for err_type, metric_name in err_config:
            metric_kwargs = get_err_type(err_type)
            if err_type == 'TotalLoss':  # special case
                err_metrics.append(
                    ErrorRecorder.init_total_loss_metric(config, criteria)
                )
                continue
            metric_cls = ErrorRecorder.METRIC_DICT[metric_name]
            assert isinstance(metric_kwargs['name'], str)
            metric_kwargs['name'] += f'_{metric_name}'
            if metric_name == 'Loss':
                metric_kwargs['func'] = criteria
                metric_kwargs['unit'] = None
            err_metrics.append(metric_cls(**metric_kwargs))
        return ErrorRecorder(err_metrics)
