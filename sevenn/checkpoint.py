import os
import pathlib
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Union

import pandas as pd
from packaging.version import Version
from torch import Tensor
from torch import load as torch_load

import sevenn
import sevenn._const as consts
import sevenn._keys as KEY
import sevenn.scripts.backward_compatibility as compat
from sevenn import model_build
from sevenn.nn.scale import get_resolved_shift_scale
from sevenn.nn.sequential import AtomGraphSequential


def assert_atoms(atoms1, atoms2, rtol=1e-5, atol=1e-6):
    import numpy as np

    def acl(a, b, rtol=rtol, atol=atol):
        return np.allclose(a, b, rtol=rtol, atol=atol)

    assert len(atoms1) == len(atoms2)
    assert acl(atoms1.get_cell(), atoms2.get_cell())
    assert acl(atoms1.get_potential_energy(), atoms2.get_potential_energy())
    assert acl(atoms1.get_forces(), atoms2.get_forces(), rtol * 10, atol * 10)
    assert acl(
        atoms1.get_stress(voigt=False),
        atoms2.get_stress(voigt=False),
        rtol * 10,
        atol * 10,
    )
    # assert acl(atoms1.get_potential_energies(), atoms2.get_potential_energies())


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
    cp_ver = Version(config.get('version', None))
    this_ver = Version(sevenn.__version__)
    if cp_ver > this_ver:
        warnings.warn(f'The checkpoint version ({cp_ver}) is newer than this source'
                      f'({this_ver}). This may cause unexpected behaviors')

    defaults = {**consts.model_defaults(config)}
    config = compat.patch_old_config(config)  # type: ignore

    scaler = model_build.init_shift_scale(config)
    shift, scale = get_resolved_shift_scale(
        scaler, config.get(KEY.TYPE_MAP), config.get(KEY.MODAL_MAP, None)
    )
    config['shift'] = shift
    config['scale'] = scale

    for k, v in defaults.items():
        if k in config:
            continue
        if os.getenv('SEVENN_DEBUG', False):
            warnings.warn(f'{k} not in config, use default value {v}', UserWarning)
        config[k] = v

    for k, v in config.items():
        if isinstance(v, Tensor):
            config[k] = v.cpu()
    return config


def _convert_e3nn_and_cueq(stct_src, stct_dst, src_config, from_cueq):
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

    cue_only_conv_followers = [
        'convolution.f.tp.f_fx.module.c',
        'convolution.f.tp.module.module.f.module.module._f.data',
    ]
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
                stct_dst[k] = v.clone().reshape(stct_dst[k].shape)
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
                stct_dst[k] = v.clone().reshape(stct_dst[k].shape)
                flag = True
            assert flag, f'Unexpected key from linear: {k}'
        elif module_name in fc_tensor_product_module_names:
            for ignore in ignores_in_fc:
                if '.'.join([module_name, ignore]) in k:
                    flag = True
                    break
            if not flag and k == '.'.join([module_name, 'fc_tensor_product.weight']):
                updated_keys.append(k)
                stct_dst[k] = v.clone().reshape(stct_dst[k].shape)
                flag = True
            assert flag, f'Unexpected key from fc tensor product: {k}'
        else:
            # assert k in stct_dst
            updated_keys.append(k)
            stct_dst[k] = v.clone().reshape(stct_dst[k].shape)

    return stct_dst


class SevenNetCheckpoint:
    """
    Tool box for checkpoint processed from SevenNet.
    """

    def __init__(self, checkpoint_path: Union[pathlib.Path, str]):
        self._checkpoint_path = os.path.abspath(checkpoint_path)
        self._config = None
        self._epoch = None
        self._model_state_dict = None
        self._optimizer_state_dict = None
        self._scheduler_state_dict = None
        self._hash = None
        self._time = None

        self._loaded = False

    def __repr__(self) -> str:
        cfg = self.config  # just alias
        if len(cfg) == 0:
            return ''
        dct = {
            'Sevennet version': cfg.get('version', 'Not found'),
            'When': self.time,
            'Hash': self.hash,
            'Cutoff': cfg.get('cutoff'),
            'Channel': cfg.get('channel'),
            'Lmax': cfg.get('lmax'),
            'Group (parity)': 'O3' if cfg.get('is_parity') else 'SO3',
            'Interaction layers': cfg.get('num_convolution_layer'),
            'Self connection type': cfg.get('self_connection_type', 'nequip'),
            'Last epoch': self.epoch,
            'Elements': len(cfg.get('chemical_species', [])),
        }
        if cfg.get('use_modality', False):
            dct['Modality'] = ', '.join(list(cfg.get('_modal_map', {}).keys()))

        df = pd.DataFrame.from_dict([dct]).T  # type: ignore
        df.columns = ['']
        return df.to_string()

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

    @property
    def epoch(self) -> Optional[int]:
        if not self._loaded:
            self._load()
        return self._epoch

    @property
    def time(self) -> str:
        if not self._loaded:
            self._load()
        assert isinstance(self._time, str)
        return self._time

    @property
    def hash(self) -> str:
        if not self._loaded:
            self._load()
        assert isinstance(self._hash, str)
        return self._hash

    def _load(self) -> None:
        assert not self._loaded
        cp_path = self.checkpoint_path  # just alias

        cp = torch_load(cp_path, weights_only=False, map_location='cpu')
        self._config_original = cp.get('config', {})
        self._model_state_dict = cp.get('model_state_dict', {})
        self._optimizer_state_dict = cp.get('optimizer_state_dict', {})
        self._scheduler_state_dict = cp.get('scheduler_state_dict', {})
        self._epoch = cp.get('epoch', None)
        self._time = cp.get('time', 'Not found')
        self._hash = cp.get('hash', 'Not found')

        if len(self._config_original) == 0:
            warnings.warn(f'config is not found from {cp_path}')
            self._config = {}
        else:
            self._config = _config_cp_routine(self._config_original)

        if len(self._model_state_dict) == 0:
            warnings.warn(f'model_state_dict is not found from {cp_path}')

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
            state_dict = _convert_e3nn_and_cueq(
                stct_src, model.state_dict(), self.config, from_cueq=cp_using_cue
            )

        missing, not_used = model.load_state_dict(state_dict, strict=False)
        if len(not_used) > 0:
            warnings.warn(f'Some keys are not used: {not_used}', UserWarning)

        assert len(missing) == 0, f'Missing keys: {missing}'
        return model

    def yaml_dict(self, mode: str) -> dict:
        """
        Return dict for input.yaml from checkpoint config
        Dataset paths and statistic values are removed intentionally
        """
        if mode not in ['reproduce', 'continue', 'continue_modal']:
            raise ValueError(f'Unknown mode: {mode}')

        ignore = [
            'when',
            KEY.DDP_BACKEND,
            KEY.LOCAL_RANK,
            KEY.IS_DDP,
            KEY.DEVICE,
            KEY.MODEL_TYPE,
            KEY.SHIFT,
            KEY.SCALE,
            KEY.CONV_DENOMINATOR,
            KEY.SAVE_DATASET,
            KEY.SAVE_BY_LABEL,
            KEY.SAVE_BY_TRAIN_VALID,
            KEY.CONTINUE,
            KEY.LOAD_DATASET,  # old
        ]

        cfg = self.config
        len_atoms = len(cfg[KEY.TYPE_MAP])

        world_size = cfg.pop(KEY.WORLD_SIZE, 1)
        cfg[KEY.BATCH_SIZE] = cfg[KEY.BATCH_SIZE] * world_size
        cfg[KEY.LOAD_TRAINSET] = '**path_to_training_set**'

        major, minor, _ = cfg.pop('version', '0.0.0').split('.')[:3]
        if int(major) == 0 and int(minor) <= 9:
            warnings.warn('checkpoint version too old, yaml may wrong')

        ret = {'model': {}, 'train': {}, 'data': {}}
        for k, v in cfg.items():
            if k.startswith('_') or k in ignore or k.endswith('set_path'):
                continue
            if k in consts.DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG:
                ret['model'][k] = v
            elif k in consts.DEFAULT_TRAINING_CONFIG:
                ret['train'][k] = v
            elif k in consts.DEFAULT_DATA_CONFIG:
                ret['data'][k] = v

        ret['model'][KEY.CHEMICAL_SPECIES] = (
            'univ' if len_atoms == consts.NUM_UNIV_ELEMENT else 'auto'
        )
        ret['data'][KEY.LOAD_TRAINSET] = '**path_to_trainset**'
        ret['data'][KEY.LOAD_VALIDSET] = '**path_to_validset**'

        # TODO
        ret['data'][KEY.SHIFT] = '**failed to infer shift, should be set**'
        ret['data'][KEY.SCALE] = '**failed to infer scale, should be set**'

        if mode.startswith('continue'):
            ret['train'].update(
                {KEY.CONTINUE: {KEY.CHECKPOINT: self.checkpoint_path}}
            )
        modal_names = None
        if mode == 'continue_modal' and not cfg.get(KEY.USE_MODALITY, False):
            ret['train'][KEY.USE_MODALITY] = True

            # suggest defaults
            ret['model'][KEY.USE_MODAL_NODE_EMBEDDING] = False
            ret['model'][KEY.USE_MODAL_SELF_INTER_INTRO] = True
            ret['model'][KEY.USE_MODAL_SELF_INTER_OUTRO] = True
            ret['model'][KEY.USE_MODAL_OUTPUT_BLOCK] = True

            ret['data'][KEY.USE_MODAL_WISE_SHIFT] = True
            ret['data'][KEY.USE_MODAL_WISE_SCALE] = False

            modal_names = ['my_modal1', 'my_modal2']
        elif cfg.get(KEY.USE_MODALITY, False):
            modal_names = list(cfg[KEY.MODAL_MAP].keys())

        if modal_names:
            ret['data'][KEY.LOAD_TRAINSET] = [
                {'data_modality': mm, 'file_list': [{'file': f'**path_to_{mm}**'}]}
                for mm in modal_names
            ]

        return ret

    def append_modal(
        self,
        dst_config,
        original_modal_name: str = 'origin',
        working_dir: str = os.getcwd(),
    ):
        """ """
        import sevenn.train.modal_dataset as modal_dataset
        from sevenn.model_build import init_shift_scale
        from sevenn.scripts.convert_model_modality import _append_modal_weight

        src_config = self.config
        src_has_no_modal = not src_config.get(KEY.USE_MODALITY, False)

        # inherit element things first
        chem_keys = [
            KEY.TYPE_MAP,
            KEY.NUM_SPECIES,
            KEY.CHEMICAL_SPECIES,
            KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER,
        ]
        dst_config.update({k: src_config[k] for k in chem_keys})

        if dst_config[KEY.USE_MODAL_WISE_SHIFT] and (
            KEY.SHIFT not in dst_config or not isinstance(dst_config[KEY.SHIFT], str)
        ):
            raise ValueError('To use modal wise shift, keyword shift is required')
        if dst_config[KEY.USE_MODAL_WISE_SCALE] and (
            KEY.SCALE not in dst_config or not isinstance(dst_config[KEY.SCALE], str)
        ):
            raise ValueError('To use modal wise scale, keyword scale is required')

        if src_has_no_modal and not dst_config[KEY.USE_MODAL_WISE_SHIFT]:
            dst_config[KEY.SHIFT] = src_config[KEY.SHIFT]
        if src_has_no_modal and not dst_config[KEY.USE_MODAL_WISE_SCALE]:
            dst_config[KEY.SCALE] = src_config[KEY.SCALE]

        # get statistics of given datasets of yaml
        # dst_config updated
        _ = modal_dataset.from_config(dst_config, working_dir=working_dir)
        dst_modal_map = dst_config[KEY.MODAL_MAP]

        found_modal_names = list(dst_modal_map.keys())
        if len(found_modal_names) == 0:
            raise ValueError('No modality is found from config')

        # Check difference btw given modals and new modal map
        orig_modal_map = src_config.get(KEY.MODAL_MAP, {original_modal_name: 0})
        assert isinstance(orig_modal_map, dict)
        new_modal_map = orig_modal_map.copy()
        for modal_name in found_modal_names:
            if modal_name in orig_modal_map:  # duplicate, skipping
                continue
            new_modal_map[modal_name] = len(new_modal_map)  # assign new
        print(f'New modals: {list(new_modal_map.keys())}')

        if src_has_no_modal:
            append_num = len(new_modal_map)
        else:
            append_num = len(new_modal_map) - len(orig_modal_map)
        if append_num == 0:
            raise ValueError('Nothing to append from checkpoint')

        dst_config[KEY.NUM_MODALITIES] = len(new_modal_map)
        dst_config[KEY.MODAL_MAP] = new_modal_map

        # update dst_config's shift scales based on src_config
        for ss_key, use_mw in (
            (KEY.SHIFT, dst_config[KEY.USE_MODAL_WISE_SHIFT]),
            (KEY.SCALE, dst_config[KEY.USE_MODAL_WISE_SCALE]),
        ):
            if not use_mw:  # not using mw ss, just assign
                assert not isinstance(dst_config[ss_key], dict)
                dst_config[ss_key] = src_config[ss_key]
            elif src_has_no_modal:
                assert isinstance(dst_config[ss_key], dict)
                # mw ss, update by dict but use original_modal_name
                dst_config[ss_key].update({original_modal_name: src_config[ss_key]})
            else:
                assert isinstance(dst_config[ss_key], dict)
                # mw ss, update by dict
                dst_config[ss_key].update(src_config[ss_key])
        scaler = init_shift_scale(dst_config)

        # finally, prepare updated continuable state dict using above
        orig_model = self.build_model()
        orig_state_dict = orig_model.state_dict()

        new_state_dict = copy_state_dict(orig_state_dict)
        for stct_key in orig_state_dict:
            sp = stct_key.split('.')
            k, follower = sp[0], '.'.join(sp[1:])
            if k == 'rescale_atomic_energy' and follower == 'shift':
                new_state_dict[stct_key] = scaler.shift.clone()
            elif k == 'rescale_atomic_energy' and follower == 'scale':
                new_state_dict[stct_key] = scaler.scale.clone()
            elif follower == 'linear.weight' and (  # append linear layer
                (
                    dst_config[KEY.USE_MODAL_NODE_EMBEDDING]
                    and k.endswith('onehot_to_feature_x')
                )
                or (
                    dst_config[KEY.USE_MODAL_SELF_INTER_INTRO]
                    and k.endswith('self_interaction_1')
                )
                or (
                    dst_config[KEY.USE_MODAL_SELF_INTER_OUTRO]
                    and k.endswith('self_interaction_2')
                )
                or (
                    dst_config[KEY.USE_MODAL_OUTPUT_BLOCK]
                    and k == 'reduce_input_to_hidden'
                )
            ):
                orig_linear = getattr(orig_model._modules[k], 'linear')
                # assert normalization element
                new_state_dict[stct_key] = _append_modal_weight(
                    orig_state_dict,
                    k,
                    orig_linear.irreps_in,
                    orig_linear.irreps_out,
                    append_num,
                )

        dst_config['version'] = sevenn.__version__

        return new_state_dict

    def get_checkpoint_dict(self) -> dict:
        """
        Return duplicate of this checkpoint with new hash and time.
        Convenient for creating variant of the checkpoint
        """
        return {
            'config': self.config,
            'epoch': self.epoch,
            'model_state_dict': self.model_state_dict,
            'optimizer_state_dict': self.optimizer_state_dict,
            'scheduler_state_dict': self.scheduler_state_dict,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'hash': uuid.uuid4().hex,
        }
