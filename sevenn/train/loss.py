from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ase.units import kB

import sevenn._const as CONST
import sevenn._keys as KEY


class LossDefinition:
    """
    Base class for loss definition
    weights are defined in outside of the class
    """

    def __init__(
        self,
        name: str,
        unit: Optional[str] = None,
        criterion: Optional[Callable] = None,
        ref_key: Optional[str] = None,
        pred_key: Optional[str] = None,
        use_weight: bool = False,
        ignore_unlabeled: bool = True,
    ) -> None:
        self.name = name
        self.unit = unit
        self.criterion = criterion
        self.ref_key = ref_key
        self.pred_key = pred_key
        self.use_weight = use_weight
        self.ignore_unlabeled = ignore_unlabeled

    def __repr__(self):
        return self.name

    def assign_criteria(self, criterion: Callable) -> None:
        if self.criterion is not None:
            raise ValueError('Loss uses its own criterion.')
        self.criterion = criterion

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.pred_key is None or self.ref_key is None:
            raise NotImplementedError('LossDefinition is not implemented.')
        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        return pred, ref, None

    def _ignore_unlabeled(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
        data_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        unlabeled = torch.isnan(ref)
        pred = pred[~unlabeled]
        ref = ref[~unlabeled]
        if data_weights is not None:
            data_weights = data_weights[~unlabeled]
        return pred, ref, data_weights

    def get_loss(self, batch_data: Dict[str, Any], model: Optional[Callable] = None):
        """
        Function that return scalar
        """
        if self.criterion is None:
            raise NotImplementedError('LossDefinition has no criterion.')
        pred, ref, w_tensor = self._preprocess(batch_data, model)

        if self.ignore_unlabeled:
            pred, ref, w_tensor = self._ignore_unlabeled(pred, ref, w_tensor)

        if len(pred) == 0:
            assert self.ref_key is not None
            return torch.zeros(1, device=batch_data[self.ref_key].device)

        loss = self.criterion(pred, ref)
        if self.use_weight:
            loss = torch.mean(loss * w_tensor)
        return loss


class PerAtomEnergyLoss(LossDefinition):
    """
    Loss for per atom energy
    """

    def __init__(
        self,
        name: str = 'Energy',
        unit: str = 'eV/atom',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.ENERGY,
        pred_key: str = KEY.PRED_TOTAL_ENERGY,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_atoms = batch_data[KEY.NUM_ATOMS]
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = batch_data[self.pred_key] / num_atoms
        ref = batch_data[self.ref_key] / num_atoms
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 1)

        return pred, ref, w_tensor


class PerAtomVibEntropyLoss(LossDefinition):
    def __init__(
        self,
        name: str = 'Entropy',
        unit: str = 'eV/atom/K',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.TOTAL_ENTROPY,
        debye_key: str = KEY.DEBYE_ENTROPY,
        pred_key: str = KEY.PRED_TOTAL_ENTROPY,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )
        self.debye_key = debye_key

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_atoms = batch_data[KEY.NUM_ATOMS]
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = batch_data[self.pred_key] / num_atoms + batch_data[self.debye_key]
        ref = batch_data[self.ref_key] / num_atoms
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 1)

        return pred, ref, w_tensor


class PerAtomVibFreeEnergyLoss(LossDefinition):
    def __init__(
        self,
        name: str = 'FreeEnergy',
        unit: str = 'eV/atom',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.TOTAL_FREE_ENERGY,
        debye_key: str = KEY.DEBYE_FREE_ENERGY,
        pred_key: str = KEY.PRED_TOTAL_FREE_ENERGY,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )
        self.debye_key = debye_key

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_atoms = batch_data[KEY.NUM_ATOMS]
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = batch_data[self.pred_key] / num_atoms + batch_data[self.debye_key]
        ref = batch_data[self.ref_key] / num_atoms
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 1)

        return pred, ref, w_tensor


class PerAtomHeatCapacityLoss(LossDefinition):
    def __init__(
        self,
        name: str = 'HeatCapacity',
        unit: str = 'eV/atom/K',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.TOTAL_HEAT_CAPACITY,
        debye_key: str = KEY.DEBYE_HEAT_CAPACITY,
        pred_key: str = KEY.PRED_TOTAL_HEAT_CAPACITY,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )
        self.debye_key = debye_key

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_atoms = batch_data[KEY.NUM_ATOMS]
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = batch_data[self.pred_key] / num_atoms + batch_data[self.debye_key]
        ref = batch_data[self.ref_key] / num_atoms
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 1)

        return pred, ref, w_tensor


class PerAtomVibAsymptotLoss(LossDefinition):
    def __init__(
        self,
        name: str = 'Asymptot',
        unit: str = 'eV*K/atom',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.TOTAL_ASYMPTOT,
        debye_key: str = KEY.DEBYE_ASYMPTOT,
        pred_key: str = KEY.PRED_TOTAL_HEAD_ASYMPTOT,
        heat_capacity_power: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )
        self.debye_key = debye_key
        self.heat_capacity_power = heat_capacity_power

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_atoms = batch_data[KEY.NUM_ATOMS]
        per_atom_heat_capacity = batch_data[KEY.PER_ATOM_HEAT_CAPACITY]
        cv_weight = (per_atom_heat_capacity / 3 / kB) ** self.heat_capacity_power  # range 0 to 1, high for converged Cv = 3 kB
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = cv_weight * (batch_data[self.pred_key] / num_atoms + batch_data[self.debye_key])
        ref = cv_weight * (batch_data[self.ref_key] / num_atoms)
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 1)

        return pred, ref, w_tensor


class ForceLoss(LossDefinition):
    """
    Loss for force
    """

    def __init__(
        self,
        name: str = 'Force',
        unit: str = 'eV/A',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.FORCE,
        pred_key: str = KEY.PRED_FORCE,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = weight[batch_data[KEY.BATCH]]
            w_tensor = torch.repeat_interleave(w_tensor, 3)

        return pred, ref, w_tensor


class StressLoss(LossDefinition):
    """
    Loss for stress this is kbar
    """

    def __init__(
        self,
        name: str = 'Stress',
        unit: str = 'kbar',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.STRESS,
        pred_key: str = KEY.PRED_STRESS,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )
        self.TO_KB = 1602.1766208  # eV/A^3 to kbar

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)

        pred = torch.reshape(batch_data[self.pred_key] * self.TO_KB, (-1,))
        ref = torch.reshape(batch_data[self.ref_key] * self.TO_KB, (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 6)

        return pred, ref, w_tensor


class L2Regularization(LossDefinition):
    """
    L2 regularization for task-specific (modal) parameters.
    Regularizes the last weight view of modal-specific IrrepsLinear layers,
    which corresponds to the modal input dimension.
    """

    def __init__(
        self,
        name: str,
        module_keys: List[str],
        reg_modal_only: bool = True,
    ):
        super().__init__(
            name=name,
            unit=None,
            criterion=None,
            ref_key=None,
            pred_key=None,
        )
        self.module_keys = module_keys
        self.reg_modal_only = reg_modal_only

    def get_loss(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ):
        device = batch_data['x'].device
        ret = torch.tensor([0.0], device=device)
        for module_key in self.module_keys:
            module = model._modules[module_key]  # type: ignore
            reg_params = list(module._modules['linear'].weight_views())[-1]
            reg_loss = torch.sum(torch.pow(reg_params, 2))
            ret = ret + reg_loss
        return ret

    def get_cosine(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ):
        cosine_list = []
        for module_key in self.module_keys:
            module = model._modules[module_key]  # type: ignore
            reg_params = list(module._modules['linear'].weight_views())[-1]
            dot = torch.dot(reg_params[0], reg_params[1])
            norm = torch.norm(reg_params[0]) * torch.norm(reg_params[1])
            cosine_list.append(dot / norm)
        ret = torch.tensor(
            [sum(cosine_list) / len(cosine_list)],
            device=batch_data['x'].device,
        )
        return ret


class TemperatureEncodingRegularization(LossDefinition):
    def __init__(
        self,
        name: str,# 'L2_T_enc'
        module_keys: List[str],
        heat_capacity_power: int=4,
    ):
        super().__init__(
            name=name,
            unit=None,
            criterion=None,
            ref_key=None,
            pred_key=None,
        )
        self.module_keys = module_keys
        self.heat_capacity_power = heat_capacity_power

    def get_loss(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ):
        encoding = batch_data[KEY.TEMPERATURE_ENC]
        per_atom_heat_capacity = batch_data[KEY.PER_ATOM_HEAT_CAPACITY]
        weight = (per_atom_heat_capacity / 3 / kB) ** self.heat_capacity_power  # range 0 to 1, high for converged Cv = 3 kB

        device = batch_data['x'].device
        ret = torch.tensor([0.0], device=device)

        for module_key in self.module_keys:
            module = model._modules[module_key]
            embedding = module.linear(encoding)
            l2_norm = torch.sum(torch.pow(embedding, 2), dim=1)
            ret = ret + torch.mean(weight * l2_norm)

        return ret


class TemperatureGateRegularization(LossDefinition):
    def __init__(
        self,
        name: str,# 'L2_T_gate'
        heat_capacity_power: int=4,
    ):
        super().__init__(
            name=name,
            unit=None,
            criterion=None,
            ref_key=None,
            pred_key=None,
        )
        self.heat_capacity_power = heat_capacity_power


    def get_loss(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ):
        per_atom_heat_capacity = batch_data[KEY.PER_ATOM_HEAT_CAPACITY]
        weight = (per_atom_heat_capacity / 3 / kB) ** self.heat_capacity_power  # range 0 to 1, high for converged Cv = 3 kB

        gate_value = batch_data['_gate'].squeeze(-1)
        reg_each_atom = 1. - gate_value
        reg_loss = torch.zeros(
            (int(batch_data[KEY.BATCH].max())+1),
            dtype = reg_each_atom.dtype,
            device = reg_each_atom.device, 
        ).scatter_reduce_(0, batch_data[KEY.BATCH], reg_each_atom, reduce='sum')
        reg_loss = reg_loss / batch_data[KEY.NUM_ATOMS]

        return torch.mean(weight * reg_loss)


def _get_modal_module_keys_for_reg(
    config: Dict[str, Any], all_module_keys: List[str]
) -> List[str]:
    module_keys_to_reg = []
    for module_key in all_module_keys:
        for (
            use_modal_module_key,
            modal_module_name,
        ) in CONST.IMPLEMENTED_MODAL_MODULE_DICT.items():
            if (
                not config[use_modal_module_key]
                or modal_module_name not in module_key
            ):
                continue
            elif modal_module_name == 'reduce_input_to_hidden':
                continue
            module_keys_to_reg.append(module_key)
    return module_keys_to_reg


def _get_temperature_block_keys_for_reg(
    config: Dict[str, Any], all_module_keys: List[str]
) -> List[str]:
    tblock_keys_to_reg = []
    for module_key in all_module_keys:
        for (
            use_tblock_module_key,
            tblock_module_name,
        ) in CONST.IMPLEMENTED_TEMPERATURE_BLOCK_DICT.items():
            if (
                not config[use_tblock_module_key]
                or tblock_module_name not in module_key
                or 'temperature_block' not in module_key
            ):
                continue
            tblock_keys_to_reg.append(module_key)
    return tblock_keys_to_reg


def get_regularization_from_config(
    config: Dict[str, Any], all_module_keys: List[str]
) -> List[Tuple[LossDefinition, float]]:
    reg_params = config.get(KEY.REG_PARAM, {})
    reg_functions: List[Tuple[LossDefinition, float]] = []

    modal_param = reg_params.get('modal', {})
    if modal_param:
        reg_weight = float(modal_param.get(KEY.REG_WEIGHT, 1e-5))
        module_keys_to_reg = _get_modal_module_keys_for_reg(
            config, all_module_keys
        )

        reg_functions.append((
            L2Regularization('L2_modal', module_keys_to_reg, reg_modal_only=True),
            reg_weight,
        ))

    t_enc_param = reg_params.get('temperature_encoding', {})
    if t_enc_param:
        reg_weight = float(t_enc_param.get(KEY.REG_WEIGHT, 1e-5))
        module_keys_to_reg = _get_temperature_block_keys_for_reg(
            config, all_module_keys
        )

        reg_functions.append((
            TemperatureEncodingRegularization('L2_T_enc', module_keys_to_reg, t_enc_param.get('heat_capacity_power', 4)),
            reg_weight,
        ))

    t_gate_param = reg_params.get('temperature_gate', {})
    if t_gate_param:
        reg_weight = float(t_gate_param.get(KEY.REG_WEIGHT, 1e-5))
        reg_functions.append((
            TemperatureGateRegularization('L2_T_gate', t_gate_param.get('heat_capacity_power', 4)),
            reg_weight,
        ))

    return reg_functions


def make_loss_info_dict_from_config(config: Dict[str, Any]):
    # this is for backward compatibility
    loss_info_dict = {}
    loss_type = config.get(KEY.LOSS, 'mse').lower()
    loss_param = config.get(KEY.LOSS_PARAM, {})
    for key in ['energy', 'force', 'stress']:
        loss_info_dict[key] = {}
        # loss_weight not initialized here.
        loss_info_dict[key].update(
            {KEY.LOSS_TYPE: loss_type, KEY.LOSS_PARAM: loss_param}
        )

    return loss_info_dict


def get_loss_functions_from_config(
    config: Dict[str, Any]
) -> List[Tuple[LossDefinition, float]]:
    from sevenn.train.optim import loss_dict

    if config.get(KEY.USE_TEMPERATURE, False):
        return get_vib_loss_func_from_config(config)

    loss_functions = []  # list of tuples (loss_definition, weight)

    loss_info_dict = config.get(KEY.LOSS, 'mse')
    if isinstance(loss_info_dict, str):
        loss_info_dict = make_loss_info_dict_from_config(config)

    loss_function_cls_dict = {
        'energy': PerAtomEnergyLoss,
        'force': ForceLoss,
        'stress': StressLoss,
    }
    loss_weights = {
        'energy': config.get(KEY.ENERGY_WEIGHT, 1.0),
        'force': config[KEY.FORCE_WEIGHT],
        'stress': config[KEY.STRESS_WEIGHT],
    }

    use_weight = config.get(KEY.USE_WEIGHT, False)
    commons = {'use_weight': use_weight}

    keys = ['energy', 'force']
    if config[KEY.IS_TRAIN_STRESS]:
        keys += ['stress']

    for key in keys:
        loss_info = loss_info_dict.get(key, {})
        loss_param = loss_info.get(KEY.LOSS_PARAM, {})
        loss_weight = loss_info.get(KEY.LOSS_WEIGHT, loss_weights[key])
        if (loss_type := loss_info.get(KEY.LOSS_TYPE, 'mse').lower()) == 'l2mae':
            if key == 'energy':
                raise NotImplementedError('L2MAE not implemented for energy.')
            else:
                loss_param.update({'prop': key})

        loss_cls = loss_dict[loss_type]
        if use_weight:
            loss_param['reduction'] = 'none'
        criterion = loss_cls(**loss_param)
        loss_function_cls = loss_function_cls_dict[key]
        loss_function = loss_function_cls(criterion=criterion, **commons)
        loss_functions.append((loss_function, loss_weight))

    return loss_functions


def get_vib_loss_func_from_config(config):
    from sevenn.train.optim import loss_dict
    loss_functions = []
    loss_info_dict = config[KEY.LOSS]

    loss_function_cls_dict = {
        'entropy': PerAtomVibEntropyLoss,
        'free_energy': PerAtomVibFreeEnergyLoss,
        'heat_capacity': PerAtomHeatCapacityLoss,
        'asymptot': PerAtomVibAsymptotLoss,
    }
    loss_weights = {
        'entropy': config.get(KEY.ENTROPY_WEIGHT, 1000.0),
        'free_energy': config.get(KEY.FREE_ENERGY_WEIGHT, 1.0),
        'heat_capacity': config.get(KEY.HEAT_CAPACITY_WEIGHT, 1000.0),
        'asymptot': config.get(KEY.ASYMPTOT_WEIGHT, 1.0),
    }

    use_weight = config.get(KEY.USE_WEIGHT, False)
    commons = {'use_weight': use_weight}

    keys = ['entropy', 'free_energy']
    if config[KEY.IS_TRAIN_HEAT_CAPACITY]:
        keys += ['heat_capacity']
    if config[KEY.IS_TRAIN_ASYMPTOT]:
        keys += ['asymptot']

    for key in keys:
        loss_info = loss_info_dict.get(key, {})
        loss_param = loss_info.get(KEY.LOSS_PARAM, {})
        loss_weight = loss_info.get(KEY.LOSS_WEIGHT, loss_weights[key])
        if (loss_type := loss_info.get(KEY.LOSS_TYPE, 'mse').lower()) == 'l2mae':
            raise NotImplementedError('L2MAE not implemented for ThreeNet.')

        loss_cls = loss_dict[loss_type]
        if use_weight:
            loss_param['reduction'] = 'none'
        criterion = loss_cls(**loss_param)
        loss_function_cls = loss_function_cls_dict[key]
        loss_function = loss_function_cls(criterion=criterion, **commons)
        loss_functions.append((loss_function, loss_weight))

    return loss_functions

