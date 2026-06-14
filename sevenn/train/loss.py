from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

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


class EWCLoss(LossDefinition):
    """
    Elastic Weight Consolidation penalty: sum_i F_i (theta_i - theta*_i)^2,
    with precomputed Fisher information F and reference parameters theta*.
    Consumes precomputed Fisher/optimal-params dicts; it does not compute them.
    """

    def __init__(
        self,
        fisher_dict: Dict[str, torch.Tensor],
        opt_params_dict: Dict[str, torch.Tensor],
        name: str = 'EWC',
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not isinstance(fisher_dict, dict) or not isinstance(
            opt_params_dict, dict
        ):
            raise ValueError('EWC fisher_information/opt_params must be dicts')
        super().__init__(name=name, use_weight=False, **kwargs)
        self.fisher_dict = fisher_dict
        self.opt_params_dict = opt_params_dict
        self._checked = False
        if device is not None:
            self.to(device)

    def to(self, device) -> None:
        self.fisher_dict = {k: v.to(device) for k, v in self.fisher_dict.items()}
        self.opt_params_dict = {
            k: v.to(device) for k, v in self.opt_params_dict.items()
        }

    def _check_and_align(self, model: Callable) -> None:
        if len(self.fisher_dict) == 0 or len(self.opt_params_dict) == 0:
            raise ValueError('EWC fisher_information/opt_params is empty')
        model_params = {
            n: p for n, p in model.named_parameters() if p.requires_grad
        }
        if len(model_params) == 0:
            raise ValueError('EWC requires the model to have trainable parameters')
        if len(set(self.fisher_dict) & set(model_params)) == 0:
            raise ValueError(
                'EWC fisher/opt_params parameter names do not match the model; '
                'the pickle was likely produced by an incompatible SevenNet '
                f'version. example model param: {next(iter(model_params))}; '
                f'example fisher key: {next(iter(self.fisher_dict))}'
            )
        # every trainable parameter must be covered by Fisher and reference
        # params with the right shape, so EWC never silently skips a parameter
        # it should constrain.
        for name, param in model_params.items():
            if name not in self.fisher_dict:
                raise ValueError(
                    f'EWC fisher_information is missing trainable param {name}'
                )
            if name not in self.opt_params_dict:
                raise ValueError(
                    f'EWC opt_params is missing trainable param {name}'
                )
            if self.fisher_dict[name].shape != param.shape:
                raise ValueError(
                    f'EWC fisher shape mismatch for {name}: '
                    f'{tuple(self.fisher_dict[name].shape)} != {tuple(param.shape)}'
                )
            if self.opt_params_dict[name].shape != param.shape:
                raise ValueError(
                    f'EWC opt_params shape mismatch for {name}: '
                    f'{tuple(self.opt_params_dict[name].shape)} != '
                    f'{tuple(param.shape)}'
                )
        self.to(next(iter(model_params.values())).device)
        self._checked = True

    def get_loss(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ):
        _ = batch_data
        if model is None:
            raise ValueError('EWCLoss requires the model to compute the penalty')
        if not self._checked:
            self._check_and_align(model)
        device = next(model.parameters()).device
        ewc_loss = torch.zeros(1, device=device)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.fisher_dict or name not in self.opt_params_dict:
                continue
            fisher = self.fisher_dict[name]
            opt_param = self.opt_params_dict[name]
            ewc_loss = ewc_loss + torch.sum(fisher * (param - opt_param) ** 2)
        return ewc_loss


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

    # reEWC: add the EWC penalty as an extra loss term when a precomputed Fisher
    # information and reference parameters are given under continue.
    cont = config.get(KEY.CONTINUE, {})
    fisher_path = cont.get(KEY.FISHER, False)
    opt_path = cont.get(KEY.OPT_PARAMS, False)
    ewc_lambda = cont.get(KEY.EWC_LAMBDA, 0)
    if fisher_path or opt_path or ewc_lambda:
        if not (fisher_path and opt_path):
            raise ValueError(
                'EWC requires both continue.fisher_information and '
                'continue.opt_params to be set'
            )
        if not (
            isinstance(ewc_lambda, (int, float))
            and not isinstance(ewc_lambda, bool)
            and ewc_lambda > 0
        ):
            raise ValueError('EWC requires continue.ewc_lambda > 0')
        fisher = torch.load(fisher_path, map_location='cpu', weights_only=True)
        opt = torch.load(opt_path, map_location='cpu', weights_only=True)
        loss_functions.append((EWCLoss(fisher, opt), ewc_lambda / 2.0))

    return loss_functions
