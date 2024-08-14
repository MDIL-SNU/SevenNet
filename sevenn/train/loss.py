import torch

import sevenn._keys as KEY


class LossDefinition:
    """
    Base class for loss definition
    weights are defined in outside of the class
    """

    def __init__(
        self,
        name=None,
        unit=None,
        criterion=None,
        ref_key=None,
        pred_key=None,
        use_weight=False,
        delete_unlabled=True,
    ):
        self.name = name
        self.unit = unit
        self.criterion = criterion
        self.ref_key = ref_key
        self.pred_key = pred_key
        self.use_weight = use_weight
        self.delete_unlabled = delete_unlabled

    def __repr__(self):
        return self.name

    def assign_criteria(self, criterion):
        if self.criterion is not None:
            raise ValueError('Loss uses its own criterion.')
        self.criterion = criterion

    def _preprocess(self, batch_data, model=None):
        if self.pred_key is None or self.ref_key is None:
            raise NotImplementedError('LossDefinition is not implemented.')
        return torch.reshape(batch_data[self.pred_key], (-1,)), torch.reshape(
            batch_data[self.ref_key], (-1,)
        )

    def get_loss(self, batch_data, model=None):
        """
        Function that return scalar
        """
        pred, ref = self._preprocess(batch_data, model)
        if self.use_weight:
            vdim = {'energy': 1, 'force': 3, 'stress': 6}
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            weight_tensor = (
                weight[batch_data[KEY.BATCH]]
                if loss_type == "force"
                else weight
            )
            weight_tensor = torch.repeat_interleave(weight_tensor, vdim[loss_type])

        if self.delete_unlabled:
            #  nan in `pred`` should be deleted in other process
            unlabled_idx = torch.isnan(ref)
            pred = pred[~unlabled_idx]
            ref = ref[~unlabled_idx]

            if self.use_weight:
                weight_tensor = weight_tensor[~unlabled_idx]

        if len(pred) > 0:  # more than one ref value is labelled.
            loss = self.criterion(pred, ref)
            if self.use_weight:
                loss = torch.mean(loss*weight_tensor)
        else:
            loss = None

        return loss


class PerAtomEnergyLoss(LossDefinition):
    """
    Loss for per atom energy
    """

    def __init__(
        self,
        name='Energy',
        unit='eV/atom',
        criterion=None,
        ref_key=KEY.ENERGY,
        pred_key=KEY.PRED_TOTAL_ENERGY,
        use_weight=False,
        delete_unlabled=True,
    ):
        super().__init__(
            name=name,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            use_weight = use_weight,
            delete_unlabled = delete_unlabled
        )

    def _preprocess(self, batch_data, model=None):
        num_atoms = batch_data[KEY.NUM_ATOMS]
        return (
            batch_data[self.pred_key] / num_atoms,
            batch_data[self.ref_key] / num_atoms,
        )


class ForceLoss(LossDefinition):
    """
    Loss for force
    """

    def __init__(
        self,
        name='Force',
        unit='eV/A',
        criterion=None,
        ref_key=KEY.FORCE,
        pred_key=KEY.PRED_FORCE,
        use_weight=False,
        delete_unlabled=True,
    ):
        super().__init__(
            name=name,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            use_weight = use_weight,
            delete_unlabled = delete_unlabled
        )

    def _preprocess(self, batch_data, model=None):
        return torch.reshape(batch_data[self.pred_key], (-1,)), torch.reshape(
            batch_data[self.ref_key], (-1,)
        )


class StressLoss(LossDefinition):
    """
    Loss for stress this is kbar
    """

    def __init__(
        self,
        name='Stress',
        unit='kbar',
        criterion=None,
        ref_key=KEY.STRESS,
        pred_key=KEY.PRED_STRESS,
        use_weight=False,
        delete_unlabled=True,
    ):
        super().__init__(
            name=name,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            use_weight = use_weight,
            delete_unlabled = delete_unlabled
        )

    def _preprocess(self, batch_data, model=None):
        TO_KB = 1602.1766208  # eV/A^3 to kbar
        return torch.reshape(
            batch_data[self.pred_key] * TO_KB, (-1,)
        ), torch.reshape(batch_data[self.ref_key] * TO_KB, (-1,))


def get_loss_functions_from_config(config):
    from sevenn.train.optim import loss_dict

    loss_functions = []  # list of tuples (loss_definition, weight)

    loss = loss_dict[config[KEY.LOSS].lower()]
    try:
        loss_param = config[KEY.LOSS_PARAM]
    except KeyError:
        loss_param = {}
    if config[KEY.USE_WEIGHT]:
        loss_param['reduction'] = 'none'
    criterion = loss(**loss_param)

    use_weight = config[KEY.USE_WEIGHT]

    loss_functions.append((PerAtomEnergyLoss(use_weight=use_weight), 1.0))
    loss_functions.append((ForceLoss(use_weight=use_weight), config[KEY.FORCE_WEIGHT]))
    if config[KEY.IS_TRAIN_STRESS]:
        loss_functions.append((StressLoss(use_weight=use_weight), config[KEY.STRESS_WEIGHT]))

    for loss_function, _ in loss_functions:
        if loss_function.criterion is None:
            loss_function.assign_criteria(criterion)

    return loss_functions
