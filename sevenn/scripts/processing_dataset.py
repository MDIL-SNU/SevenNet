import os

import torch
import torch.distributed as dist

import sevenn._const as CONST
import sevenn._keys as KEY
from sevenn.logger import Logger
from sevenn.train.dataload import file_to_dataset, match_reader
from sevenn.train.dataset import AtomGraphDataset
from sevenn.util import chemical_species_preprocess, onehot_to_chem


def dataset_load(file: str, config):
    """
    Wrapping of dataload.file_to_dataset to suppert
    graph prebuilt sevenn_data
    """
    log = Logger()
    log.write(f'Loading {file}\n')
    log.timer_start('loading dataset')

    if file.endswith('.sevenn_data'):
        dataset = torch.load(file, map_location='cpu', weights_only=False)
    else:
        reader, _ = match_reader(
            config[KEY.DATA_FORMAT], **config[KEY.DATA_FORMAT_ARGS]
        )
        dataset = file_to_dataset(
            file,
            config[KEY.CUTOFF],
            config[KEY.PREPROCESS_NUM_CORES],
            reader=reader,
            use_modality=config[KEY.USE_MODALITY],
            use_weight=config[KEY.USE_WEIGHT],
        )
    log.format_k_v('loaded dataset size is', dataset.len(), write=True)
    log.timer_end('loading dataset', 'data set loading time')
    return dataset


def calculate_shift_or_scale_from_key(
    train_set: AtomGraphDataset, key_given, n_chem
):
    _expand = True
    use_species_wise_shift_scale = False
    if key_given == 'per_atom_energy_mean':
        shift_or_scale = train_set.get_per_atom_energy_mean()
    elif key_given == 'elemwise_reference_energies':
        shift_or_scale = train_set.get_species_ref_energy_by_linear_comb(n_chem)
        _expand = False
        use_species_wise_shift_scale = True

    elif key_given == 'force_rms':
        shift_or_scale = train_set.get_force_rms()
    elif key_given == 'per_atom_energy_std':
        shift_or_scale = train_set.get_statistics(KEY.PER_ATOM_ENERGY)['Total'][
            'std'
        ]
    elif key_given == 'elemwise_force_rms':
        shift_or_scale = train_set.get_species_wise_force_rms(n_chem)
        _expand = False
        use_species_wise_shift_scale = True

    return shift_or_scale, _expand, use_species_wise_shift_scale


def handle_shift_scale(config, train_set: AtomGraphDataset, checkpoint_given):
    """
    Priority (first comes later to overwrite):
        1. Float given in yaml
        2. Use statistic values of checkpoint == True
        3. Plain options (provided as string)
    """
    log = Logger()
    shift, scale, conv_denominator = None, None, None
    type_map = config[KEY.TYPE_MAP]
    n_chem = len(type_map)
    chem_strs = onehot_to_chem(list(range(n_chem)), type_map)

    log.writeline('\nCalculating statistic values from dataset')

    shift_given = config[KEY.SHIFT]
    scale_given = config[KEY.SCALE]
    _expand_shift = True
    _expand_scale = True
    use_species_wise_shift = False
    use_species_wise_scale = False

    use_modal_wise_shift = config[KEY.USE_MODAL_WISE_SHIFT]
    use_modal_wise_scale = config[KEY.USE_MODAL_WISE_SCALE]

    if shift_given in CONST.IMPLEMENTED_SHIFT:
        shift, _expand_shift, use_species_wise_shift = (
            calculate_shift_or_scale_from_key(train_set, shift_given, n_chem)
        )

    if scale_given in CONST.IMPLEMENTED_SCALE:
        scale, _expand_scale, use_species_wise_scale = (
            calculate_shift_or_scale_from_key(train_set, scale_given, n_chem)
        )

    if use_modal_wise_shift or use_modal_wise_scale:
        atomdata_dict_sort_by_modal = train_set.get_dict_sort_by_modality()
        modal_map = config[KEY.MODAL_MAP]
        n_modal = len(modal_map)
        cutoff = config[KEY.CUTOFF]

        if use_modal_wise_shift:
            shift = torch.zeros((n_modal, n_chem))

        if use_modal_wise_scale:
            scale = torch.zeros((n_modal, n_chem))

        for modal_key, data_list in atomdata_dict_sort_by_modal.items():
            modal_set = AtomGraphDataset(data_list, cutoff, x_is_one_hot_idx=True)

            if use_modal_wise_shift:
                if shift_given == 'elemwise_reference_energies':
                    modal_shift, _expand_shift, use_species_wise_shift = (
                        calculate_shift_or_scale_from_key(
                            modal_set, shift_given, n_chem
                        )
                    )
                    shift[modal_map[modal_key]] = torch.tensor(
                        modal_shift
                    )  # this is np.array
                elif shift_given in CONST.IMPLEMENTED_SHIFT:
                    raise NotImplementedError(
                        'Currently, modal-wise shift implemented for'
                        'species-dependent case only.'
                    )

            if use_modal_wise_scale:
                if scale_given == 'elemwise_force_rms':
                    modal_scale, _expand_scale, use_species_wise_scale = (
                        calculate_shift_or_scale_from_key(
                            modal_set, scale_given, n_chem
                        )
                    )
                    scale[modal_map[modal_key]] = modal_scale
                elif scale_given in CONST.IMPLEMENTED_SCALE:
                    raise NotImplementedError(
                        'Currently, modal-wise scale implemented for'
                        'species-dependent case only.'
                    )

    avg_num_neigh = train_set.get_avg_num_neigh()
    log.format_k_v('Average # of neighbors', f'{avg_num_neigh:.6f}', write=True)

    if config[KEY.CONV_DENOMINATOR] == 'avg_num_neigh':
        conv_denominator = avg_num_neigh
    elif config[KEY.CONV_DENOMINATOR] == 'sqrt_avg_num_neigh':
        conv_denominator = avg_num_neigh ** (0.5)

    if (
        checkpoint_given
        and config[KEY.CONTINUE][KEY.USE_STATISTIC_VALUES_OF_CHECKPOINT]
    ):
        log.writeline(
            'Overwrite shift, scale, conv_denominator from model checkpoint'
        )
        # TODO: This needs refactoring
        conv_denominator = config[KEY.CONV_DENOMINATOR + '_cp']
        if not (use_modal_wise_shift or use_modal_wise_scale):
            # Values extracted from checkpoint in processing_continue.py
            if len(list(shift)) > 1:
                use_species_wise_shift = True
                use_species_wise_scale = True
                _expand_shift = _expand_scale = False
            else:
                shift = shift.item()
                scale = scale.item()
        else:
            # Case of modal wise shift scale
            shift_cp = config[KEY.SHIFT + '_cp']
            scale_cp = config[KEY.SCALE + '_cp']
            if not use_modal_wise_shift:
                shift = shift_cp
            if not use_modal_wise_scale:
                scale = scale_cp
            modal_map = config[KEY.MODAL_MAP]
            modal_map_cp = config[KEY.MODAL_MAP + '_cp']

            # Extracting shift, scale for modal in checkpoint model.
            if config[KEY.USE_MODALITY + '_cp']:  # cp model is multimodal
                for modal_key_cp, modal_idx_cp in modal_map_cp.items():
                    modal_idx = modal_map[modal_key_cp]
                    if use_modal_wise_shift:
                        shift[modal_idx] = torch.tensor(shift_cp[modal_idx_cp])
                    if use_modal_wise_scale:
                        scale[modal_idx] = torch.tensor(scale_cp[modal_idx_cp])

            else:  # cp model is single modal
                try:
                    modal_idx = modal_map[config[KEY.DEFAULT_MODAL]]
                except:
                    raise KeyError(
                        f'{config[KEY.DEFAULT_MODAL]} should be one of'
                        f' {modal_map.keys()}'
                    )
                if use_modal_wise_shift:
                    shift[modal_idx] = torch.tensor(shift_cp)
                if use_modal_wise_scale:
                    scale[modal_idx] = torch.tensor(scale_cp)

            if not config[KEY.CONTINUE][KEY.USE_STATISTIC_VALUES_FOR_CP_MODAL_ONLY]:
                # Also overwrite values of new modal to reference value
                # For multimodal, set reference modal with KEY.DEFAULT_MODAL
                shift_ref = shift_cp
                scale_ref = scale_cp
                if config[KEY.USE_MODALITY + '_cp']:
                    try:
                        modal_idx_cp = modal_map_cp[config[KEY.DEFAULT_MODAL]]
                    except:
                        raise KeyError(
                            f'{config[KEY.DEFAULT_MODAL]} should be one of'
                            f' {modal_map_cp.keys()}'
                        )
                    shift_ref = shift_cp[modal_idx_cp]
                    scale_ref = scale_cp[modal_idx_cp]

                for modal_key, modal_idx in modal_map.items():
                    if modal_key not in modal_map_cp.keys():
                        if use_modal_wise_shift:
                            shift[modal_idx] = shift_ref
                        if use_modal_wise_scale:
                            scale[modal_idx] = scale_ref

    # overwrite shift scale anyway if defined in yaml.
    if type(shift_given) in [list, float]:
        log.writeline('Overwrite shift to value(s) given in yaml')
        _expand_shift = isinstance(shift_given, float)
        shift = shift_given
    if type(scale_given) in [list, float]:
        log.writeline('Overwrite scale to value(s) given in yaml')
        _expand_scale = isinstance(scale_given, float)
        scale = scale_given

    if isinstance(config[KEY.CONV_DENOMINATOR], float):
        log.writeline('Overwrite conv_denominator to value given in yaml')
        conv_denominator = config[KEY.CONV_DENOMINATOR]

    if isinstance(conv_denominator, float):
        conv_denominator = [conv_denominator] * config[KEY.NUM_CONVOLUTION]

    use_species_wise_shift_scale = use_species_wise_shift or use_species_wise_scale
    if use_species_wise_shift_scale:
        chem_strs = onehot_to_chem(list(range(n_chem)), type_map)
        if _expand_shift:
            if use_modal_wise_shift:
                shift = torch.full((n_modal, n_chem), shift)
            else:
                shift = [shift] * n_chem
        if _expand_scale:
            if use_modal_wise_scale:
                scale = torch.full((n_modal, n_chem), scale)
            else:
                scale = [scale] * n_chem

        Logger().write('Use element-wise shift, scale\n')
        if use_modal_wise_shift or use_modal_wise_scale:
            for modal_key, modal_idx in modal_map.items():
                Logger().writeline(f'For modal = {modal_key}')
                print_shift = shift[modal_idx] if use_modal_wise_shift else shift
                print_scale = scale[modal_idx] if use_modal_wise_scale else scale
                for cstr, sh, sc in zip(chem_strs, print_shift, print_scale):
                    Logger().format_k_v(f'{cstr}', f'{sh:.6f}, {sc:.6f}', write=True)
        else:
            for cstr, sh, sc in zip(chem_strs, shift, scale):
                Logger().format_k_v(f'{cstr}', f'{sh:.6f}, {sc:.6f}', write=True)
    else:
        log.write('Use global shift, scale\n')
        log.format_k_v('shift, scale', f'{shift:.6f}, {scale:.6f}', write=True)

    assert isinstance(conv_denominator, list) and all(
        isinstance(deno, float) for deno in conv_denominator
    )
    log.format_k_v(
        '(1st) conv_denominator is', f'{conv_denominator[0]:.6f}', write=True
    )

    config[KEY.USE_SPECIES_WISE_SHIFT_SCALE] = use_species_wise_shift_scale
    return shift, scale, conv_denominator


# TODO: This is too long
def processing_dataset(config, working_dir):
    log = Logger()
    prefix = f'{os.path.abspath(working_dir)}/'
    is_stress = config[KEY.IS_TRAIN_STRESS]
    checkpoint_given = config[KEY.CONTINUE][KEY.CHECKPOINT] is not False
    cutoff = config[KEY.CUTOFF]

    log.write('\nInitializing dataset...\n')

    dataset = AtomGraphDataset({}, cutoff)
    load_dataset = config[KEY.LOAD_DATASET]
    if type(load_dataset) is str:
        load_dataset = [load_dataset]
    for file in load_dataset:
        dataset.augment(dataset_load(file, config))

    dataset.group_by_key()  # apply labels inside original datapoint
    dataset.unify_dtypes()  # unify dtypes of all data points

    # TODO: I think manual chemical species input is redundant
    chem_in_db = dataset.get_species()
    if config[KEY.CHEMICAL_SPECIES] == 'auto' and not checkpoint_given:
        log.writeline('Auto detect chemical species from dataset')
        config.update(chemical_species_preprocess(chem_in_db))
    elif config[KEY.CHEMICAL_SPECIES] == 'auto' and checkpoint_given:
        pass  # copied from checkpoint in processing_continue.py
    elif config[KEY.CHEMICAL_SPECIES] != 'auto' and not checkpoint_given:
        pass  # processed in parse_input.py
    else:  # config[KEY.CHEMICAL_SPECIES] != "auto" and checkpoint_given
        log.writeline('Ignore chemical species in yaml, use checkpoint')
        # already processed in processing_continue.py

    # basic dataset compatibility check with previous model
    if checkpoint_given:
        chem_from_cp = config[KEY.CHEMICAL_SPECIES]
        if not all(chem in chem_from_cp for chem in chem_in_db):
            raise ValueError('Chemical species in checkpoint is not compatible')

    # check what modalities are used in dataset
    if config[KEY.USE_MODALITY]:
        modalities = dataset.get_modalities()
        num_modalities = len(modalities)
        if num_modalities < 2:
            Logger().writeline('Only one modal is given, ignore modality')
            config.uptate({KEY.USE_MODALITY: False})

        else:
            modal_map_cp = config[KEY.MODAL_MAP + '_cp'] if checkpoint_given else {}
            modal_map = modal_map_cp.copy()
            current_idx = len(modal_map_cp)
            for modal_key in modalities:
                if modal_key not in modal_map.keys():
                    modal_map[modal_key] = current_idx
                    current_idx += 1

            if config[KEY.IS_DDP]:
                # Synchronize modal_map
                torch.cuda.set_device(config[KEY.LOCAL_RANK])
                modal_map_bcast = [modal_map]
                dist.broadcast_object_list(modal_map_bcast, src=0)
                modal_map = modal_map_bcast[0]

            config.update(
                {
                    KEY.NUM_MODALITIES: len(modal_map),
                    KEY.MODAL_MAP: modal_map,
                    KEY.MODAL_LIST: list(modal_map.keys()),
                }
            )

            dataset.write_modal_attr(
                modal_map,
                config[KEY.USE_MODAL_WISE_SHIFT] or config[KEY.USE_MODAL_WISE_SCALE],
            )

    # --------------- save dataset regardless of train/valid--------------#
    save_dataset = config[KEY.SAVE_DATASET]
    save_by_label = config[KEY.SAVE_BY_LABEL]
    if save_dataset:
        if save_dataset.endswith('.sevenn_data') is False:
            save_dataset += '.sevenn_data'
        if (save_dataset.startswith('.') or save_dataset.startswith('/')) is False:
            save_dataset = prefix + save_dataset  # save_data set is plain file name
        dataset.save(save_dataset)
        log.format_k_v('Dataset saved to', save_dataset, write=True)
        # log.write(f"Loaded full dataset saved to : {save_dataset}\n")
    if save_by_label:
        dataset.save(prefix, by_label=True)
        log.format_k_v('Dataset saved by label', prefix, write=True)
    # --------------------------------------------------------------------#

    # TODO: testset is not used
    ignore_test = not config.get(KEY.USE_TESTSET, False)
    if KEY.LOAD_VALIDSET in config and config[KEY.LOAD_VALIDSET]:
        train_set = dataset
        test_set = AtomGraphDataset([], config[KEY.CUTOFF])

        log.write('Loading validset from load_validset\n')
        valid_set = AtomGraphDataset({}, cutoff)
        for file in config[KEY.LOAD_VALIDSET]:
            valid_set.augment(dataset_load(file, config))
        valid_set.group_by_key()
        valid_set.unify_dtypes()

        # condition: validset labels should be subset of trainset labels
        valid_labels = valid_set.user_labels
        train_labels = train_set.user_labels
        if set(valid_labels).issubset(set(train_labels)) is False:
            valid_set = AtomGraphDataset(valid_set.to_list(), cutoff)
            valid_set.rewrite_labels_to_data()
            train_set = AtomGraphDataset(train_set.to_list(), cutoff)
            train_set.rewrite_labels_to_data()
            Logger().write('WARNING! validset labels is not subset of trainset\n')
            Logger().write('We overwrite all the train, valid labels to default.\n')
            Logger().write('Please create validset by sevenn_graph_build with -l\n')

        Logger().write('the validset loaded, load_dataset is now train_set\n')
        Logger().write('the ratio will be ignored\n')

        # condition: validset modalities should be subset of trainset modalities
        if config[KEY.USE_MODALITY]:
            config_modality = config[KEY.MODAL_LIST]
            valid_modality = valid_set.get_modalities()

            if set(valid_modality).issubset(set(config_modality)) is False:
                raise ValueError('validset modality is not subset of trainset')

            valid_set.write_modal_attr(
                config[KEY.MODAL_MAP],
                config[KEY.USE_MODAL_WISE_SHIFT] or config[KEY.USE_MODAL_WISE_SCALE],
            )
    else:
        train_set, valid_set, test_set = dataset.divide_dataset(
            config[KEY.RATIO], ignore_test=ignore_test
        )
        log.write(f'The dataset divided into train, valid by {KEY.RATIO}\n')

    log.format_k_v('\nloaded trainset size is', train_set.len(), write=True)
    log.format_k_v('\nloaded validset size is', valid_set.len(), write=True)

    log.write('Dataset initialization was successful\n')

    log.write('\nNumber of atoms in the train_set:\n')
    log.natoms_write(train_set.get_natoms(config[KEY.TYPE_MAP]))

    log.bar()
    log.write('Per atom energy(eV/atom) distribution:\n')
    log.statistic_write(train_set.get_statistics(KEY.PER_ATOM_ENERGY))
    log.bar()
    log.write('Force(eV/Angstrom) distribution:\n')
    log.statistic_write(train_set.get_statistics(KEY.FORCE))
    log.bar()
    log.write('Stress(eV/Angstrom^3) distribution:\n')
    try:
        log.statistic_write(train_set.get_statistics(KEY.STRESS))
    except KeyError:
        log.write('\n Stress is not included in the train_set\n')
        if is_stress:
            is_stress = False
            log.write('Turn off stress training\n')
    log.bar()

    # saved data must have atomic numbers as X not one hot idx
    if config[KEY.SAVE_BY_TRAIN_VALID]:
        train_set.save(prefix + 'train')
        valid_set.save(prefix + 'valid')
        log.format_k_v('Dataset saved by train, valid', prefix, write=True)

    # inconsistent .info dict give error when collate
    _, _ = train_set.separate_info()
    _, _ = valid_set.separate_info()

    if train_set.x_is_one_hot_idx is False:
        train_set.x_to_one_hot_idx(config[KEY.TYPE_MAP])
    if valid_set.x_is_one_hot_idx is False:
        valid_set.x_to_one_hot_idx(config[KEY.TYPE_MAP])

    log.format_k_v('training_set size', train_set.len(), write=True)
    log.format_k_v('validation_set size', valid_set.len(), write=True)

    shift, scale, conv_denominator = handle_shift_scale(
        config, train_set, checkpoint_given
    )
    config.update(
        {
            KEY.SHIFT: shift,
            KEY.SCALE: scale,
            KEY.CONV_DENOMINATOR: conv_denominator,
        }
    )

    data_lists = (train_set.to_list(), valid_set.to_list(), test_set.to_list())

    return data_lists
