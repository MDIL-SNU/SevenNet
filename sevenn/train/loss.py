from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

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


def get_regularization_from_config(
    config: Dict[str, Any], all_module_keys: List[str]
) -> List[Tuple[LossDefinition, float]]:
    reg_params = config.get(KEY.REG_PARAM, {})
    reg_functions: List[Tuple[LossDefinition, float]] = []

    modal_param = reg_params.get('modal', {})
    if not modal_param:
        return reg_functions

    reg_weight = float(modal_param.get(KEY.REG_WEIGHT, 1e-5))
    module_keys_to_reg = _get_modal_module_keys_for_reg(
        config, all_module_keys
    )

    reg_functions.append((
        L2Regularization('L2_modal', module_keys_to_reg, reg_modal_only=True),
        reg_weight,
    ))

    return reg_functions


def get_loss_functions_from_config(
    config: Dict[str, Any],
) -> List[Tuple[LossDefinition, float]]:
    from sevenn.train.optim import loss_dict

    loss_functions = []  # list of tuples (loss_definition, weight)

    loss = loss_dict[config[KEY.LOSS].lower()]
    loss_param = config.get(KEY.LOSS_PARAM, {})

    use_weight = config.get(KEY.USE_WEIGHT, False)
    if use_weight:
        loss_param['reduction'] = 'none'
    criterion = loss(**loss_param)

    commons = {'use_weight': use_weight}

    loss_functions.append((PerAtomEnergyLoss(**commons), 1.0))
    loss_functions.append((ForceLoss(**commons), config[KEY.FORCE_WEIGHT]))
    if config[KEY.IS_TRAIN_STRESS]:
        loss_functions.append((StressLoss(**commons), config[KEY.STRESS_WEIGHT]))

    for loss_function, _ in loss_functions:  # why do these?
        if loss_function.criterion is None:
            loss_function.assign_criteria(criterion)

    return loss_functions
