from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import sevenn._keys as KEY
from sevenn.train.loss import LossDefinition


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


def append_ewc_loss(
    loss_functions: List[Tuple[LossDefinition, float]],
    config: Dict[str, Any],
) -> None:
    """reEWC: append the EWC penalty as an extra loss term when a precomputed
    Fisher information and reference parameters are given under continue."""
    cont = config.get(KEY.CONTINUE, {})
    fisher_path = cont.get(KEY.FISHER, False)
    opt_path = cont.get(KEY.OPT_PARAMS, False)
    ewc_lambda = cont.get(KEY.EWC_LAMBDA, 0)
    if not (fisher_path or opt_path or ewc_lambda):
        return
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
