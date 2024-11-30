import os
import pathlib
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Union

from torch import Tensor
from torch import load as torch_load

import sevenn._keys as KEY
import sevenn.scripts.backward_compatibility as compat
from sevenn._const import model_defaults
from sevenn.nn.sequential import AtomGraphSequential


def copy_state_dict(state_dict) -> dict:
    if isinstance(state_dict, dict):
        return {key: copy_state_dict(value) for key, value in state_dict.items()}
    elif isinstance(state_dict, list):
        return [copy_state_dict(item) for item in state_dict]  # type: ignore
    elif isinstance(state_dict, Tensor):
        return state_dict.clone()  # type: ignore
    else:
        # For non-tensor values (e.g., scalars, None), return as-is
        return state_dict


def _config_cp_routine(config):
    defaults = {**model_defaults(config)}
    config = compat.patch_old_config(config)  # type: ignore

    for k, v in defaults.items():
        if k in config:
            continue
        if os.getenv('SEVENN_DEBUG', False):
            warnings.warn(f'{k} not in config, use default value {v}', UserWarning)
        config[k] = v

    # expect only non-tensor values in config, if exists, move to cpu
    # This can be happen if config has torch tensor as value (shift, scale)
    # TODO: save only non-tensors at first place is better
    for k, v in config.items():
        if isinstance(v, Tensor):
            config[k] = v.cpu()
    return config


def _e3nn_to_cue(stct_src, stct_dst, src_config):
    """
    manually check keys and assert if something unexpected happens
    """
    n_layer = src_config['num_convolution_layer']

    linear_module_names = [
        'onehot_to_feature_x',
        'reduce_input_to_hidden',
        'reduce_hidden_to_energy',
    ]
    convolution_module_names = []
    fc_tensor_product_module_names = []
    for i in range(n_layer):
        linear_module_names.append(f'{i}_self_interaction_1')
        linear_module_names.append(f'{i}_self_interaction_2')
        if src_config.get(KEY.SELF_CONNECTION_TYPE) == 'linear':
            linear_module_names.append(f'{i}_self_connection_intro')
        elif src_config.get(KEY.SELF_CONNECTION_TYPE) == 'nequip':
            fc_tensor_product_module_names.append(f'{i}_self_connection_intro')
        convolution_module_names.append(f'{i}_convolution')

    # Rule: those keys can be safely ignored before state dict load,
    #       except for linear.bias. This should be aborted in advance to
    #       this function. Others are not parameters but constants.
    cue_only_linear_followers = ['linear.f.tp.f_fx.module.c']
    e3nn_only_linear_followers = ['linear.bias', 'linear.output_mask']
    ignores_in_linear = cue_only_linear_followers + e3nn_only_linear_followers

    cue_only_conv_followers = ['convolution.f.tp.f_fx.module.c']
    e3nn_only_conv_followers = [
        'convolution._compiled_main_left_right._w3j',
        'convolution.weight',
        'convolution.output_mask',
    ]
    ignores_in_conv = cue_only_conv_followers + e3nn_only_conv_followers

    cue_only_fc_followers = ['fc_tensor_product.f.tp.f_fx.module.c']
    e3nn_only_fc_followers = [
        'fc_tensor_product.output_mask',
    ]
    ignores_in_fc = cue_only_fc_followers + e3nn_only_fc_followers

    updated_keys = []
    for k, v in stct_src.items():
        module_name = k.split('.')[0]
        flag = False
        if module_name in linear_module_names:
            for ignore in ignores_in_linear:
                if '.'.join([module_name, ignore]) in k:
                    flag = True
                    break
            if not flag and k == '.'.join([module_name, 'linear.weight']):
                updated_keys.append(k)
                stct_dst[k] = v.clone()
                flag = True
            assert flag, f'Unexpected key from linear: {k}'
        elif module_name in convolution_module_names:
            for ignore in ignores_in_conv:
                if '.'.join([module_name, ignore]) in k:
                    flag = True
                    break
            if not flag and (
                k.startswith(f'{module_name}.weight_nn')
                or k == '.'.join([module_name, 'denominator'])
            ):
                updated_keys.append(k)
                stct_dst[k] = v.clone()
                flag = True
            assert flag, f'Unexpected key from linear: {k}'
        elif module_name in fc_tensor_product_module_names:
            for ignore in ignores_in_fc:
                if '.'.join([module_name, ignore]) in k:
                    flag = True
                    break
            if not flag and k == '.'.join([module_name, 'fc_tensor_product.weight']):
                updated_keys.append(k)
                stct_dst[k] = v.clone()
                flag = True
            assert flag, f'Unexpected key from fc tensor product: {k}'
        else:
            # assert k in stct_dst
            updated_keys.append(k)
            stct_dst[k] = v.clone()

    return stct_dst


class SevenNetCheckpoint:
    """
    Tool box for checkpoint processed from SevenNet.
    """

    def __init__(self, checkpoint_path: Union[pathlib.Path, str]):
        self._checkpoint_path = checkpoint_path
        self._config = None
        self._epoch = None
        self._model_state_dict = None
        self._optimizer_state_dict = None
        self._scheduler_state_dict = None

        self._loaded = False

    @property
    def checkpoint_path(self) -> str:
        return str(self._checkpoint_path)

    @property
    def config(self) -> Dict[str, Any]:
        if not self._loaded:
            self._load()
        assert isinstance(self._config, dict)
        return deepcopy(self._config)

    @property
    def epoch(self) -> Optional[int]:
        if not self._loaded:
            self._load()
        return self._epoch

    @property
    def model_state_dict(self) -> Dict[str, Any]:
        if not self._loaded:
            self._load()
        assert isinstance(self._model_state_dict, dict)
        return copy_state_dict(self._model_state_dict)

    @property
    def optimizer_state_dict(self) -> Dict[str, Any]:
        if not self._loaded:
            self._load()
        assert isinstance(self._optimizer_state_dict, dict)
        return copy_state_dict(self._optimizer_state_dict)

    @property
    def scheduler_state_dict(self) -> Dict[str, Any]:
        if not self._loaded:
            self._load()
        assert isinstance(self._scheduler_state_dict, dict)
        return copy_state_dict(self._scheduler_state_dict)

    def _load(self) -> None:
        assert not self._loaded
        cp_path = self.checkpoint_path  # just alias

        cp = torch_load(cp_path, weights_only=False, map_location='cpu')
        self._config_original = cp.get('config', {})
        self._model_state_dict = cp.get('model_state_dict', {})
        self._optimizer_state_dict = cp.get('optimizer_state_dict', {})
        self._scheduler_state_dict = cp.get('scheduler_state_dict', {})
        self._epoch = cp.get('epoch', None)

        if len(self._config_original) == 0:
            warnings.warn(f'config is not found from {cp_path}')
            self._config = {}
        else:
            self._config = _config_cp_routine(self._config_original)

        if len(self._model_state_dict) == 0:
            warnings.warn(f'model_state_dict is not found from {cp_path}')

        if len(self._optimizer_state_dict) == 0:
            warnings.warn(f'optimizer_state_dict is not found from {cp_path}')

        if len(self._scheduler_state_dict) == 0:
            warnings.warn(f'scheduler_state_dict is not found from {cp_path}')

        if self._epoch is None:
            warnings.warn(f'epoch is not found from {cp_path}')

        self._loaded = True

    def build_model(self, backend: Optional[str] = None) -> AtomGraphSequential:
        from .model_build import build_E3_equivariant_model

        use_cue = not backend or backend.lower() in ['cue', 'cueq']
        try:
            cp_using_cue = self.config[KEY.CUEQUIVARIANCE_CONFIG]['use']
        except KeyError:
            cp_using_cue = False

        if (not backend) or (use_cue == cp_using_cue):
            # backend not given, or checkpoint backend is same as requested
            model = build_E3_equivariant_model(self.config)
            state_dict = compat.patch_state_dict_if_old(
                self.model_state_dict, self.config, model
            )
        else:
            cfg_new = self.config
            cfg_new[KEY.CUEQUIVARIANCE_CONFIG] = {'use': use_cue}
            model = build_E3_equivariant_model(cfg_new)
            stct_src = compat.patch_state_dict_if_old(
                self.model_state_dict, self.config, model
            )
            state_dict = _e3nn_to_cue(stct_src, model.state_dict(), self.config)

        missing, not_used = model.load_state_dict(state_dict, strict=False)
        if len(not_used) > 0:
            warnings.warn(f'Some keys are not used: {not_used}', UserWarning)

        assert len(missing) == 0, f'Missing keys: {missing}'
        return model

    def __repr__(self) -> str:
        return ''
