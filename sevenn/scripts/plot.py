import matplotlib
import matplotlib.pyplot as plt

from sevenn.train.trainer import DataSetType, LossType

matplotlib.use('pdf')


def draw_learning_curve(loss_history, fname):
    """
    try:
        draw_learning_curve.history[DataSetType.TRAIN][LossType.ENERGY].append(
            train_loss['total'][LossType.ENERGY])
        draw_learning_curve.history[DataSetType.TRAIN][LossType.FORCE].append(
            train_loss['total'][LossType.FORCE])
        draw_learning_curve.history[DataSetType.VALID][LossType.ENERGY].append(
            valid_loss['total'][LossType.ENERGY])
        draw_learning_curve.history[DataSetType.VALID][LossType.FORCE].append(
            valid_loss['total'][LossType.FORCE])
    except AttributeError:
        draw_learning_curve.history = {
            DataSetType.TRAIN: {LossType.ENERGY: [], LossType.FORCE: []},
            DataSetType.VALID: {LossType.ENERGY: [], LossType.FORCE: []}
        }
    """

    # alias
    train_hist = loss_history[DataSetType.TRAIN]
    valid_hist = loss_history[DataSetType.VALID]
    epoch_num = len(train_hist[LossType.ENERGY])

    # do nothinig untill 10 epoch
    if epoch_num < 10:
        return
    plt.clf()

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), dpi=300, constrained_layout=True)

    energy_ax = axs[0]
    force_ax = axs[1]
    energy_ax.plot(train_hist[LossType.ENERGY], label='train')
    energy_ax.plot(valid_hist[LossType.ENERGY], label='valid')
    energy_ax.set_xlim(3, epoch_num - 1)
    energy_ax.set_xlabel('Epoch')
    energy_ax.set_ylabel('RMSE(eV/atom)')
    energy_ax.set_title('Energy')
    energy_ax.legend()

    force_ax.plot(train_hist[LossType.FORCE], label='train')
    force_ax.plot(valid_hist[LossType.FORCE], label='valid')
    force_ax.set_xlim(3, epoch_num - 1)
    force_ax.set_xlabel('Epoch')
    force_ax.set_ylabel(r'RMSE(eV/${\rm \AA}$)')
    force_ax.set_title('Force')
    force_ax.legend()

    plt.savefig(fname, format="png")


def draw_learning_curve_deprecated(loss_hist, fname):
    # do nothinig untill 10 epoch
    if len(loss_hist[DataSetType.TRAIN]['total']['energy']) < 10:
        return
    plt.clf()

    # cut unusually large loss at early epoch for visual

    # TODO: refactor
    loss_hist[DataSetType.TRAIN]['total']['energy'] = \
        loss_hist[DataSetType.TRAIN]['total']['energy'][5:]
    loss_hist[DataSetType.TRAIN]['total']['force'] = \
        loss_hist[DataSetType.TRAIN]['total']['force'][5:]
    loss_hist[DataSetType.VALID]['total']['energy'] = \
        loss_hist[DataSetType.VALID]['total']['energy'][5:]
    loss_hist[DataSetType.VALID]['total']['force'] = \
        loss_hist[DataSetType.VALID]['total']['force'][5:]

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), dpi=300, constrained_layout=True)

    energy_ax = axs[0]
    force_ax = axs[1]
    energy_ax.plot(loss_hist[DataSetType.TRAIN]['total']['energy'], label='train')
    energy_ax.plot(loss_hist[DataSetType.VALID]['total']['energy'], label='valid')
    energy_ax.set_xlabel('Epoch')
    energy_ax.set_ylabel('RMSE(eV/atom)')
    energy_ax.set_title('Energy')
    energy_ax.legend()

    force_ax.plot(loss_hist[DataSetType.TRAIN]['total']['force'], label='train')
    force_ax.plot(loss_hist[DataSetType.VALID]['total']['force'], label='valid')
    force_ax.set_xlabel('Epoch')
    force_ax.set_ylabel(r'RMSE(eV/${\rm \AA}$)')
    force_ax.set_title('Force')
    force_ax.legend()

    plt.savefig(fname, format="png")


# TODO: implement
def draw_parity(loss_hist, fname):
    pass
    """
    plt.clf()

    cmap = plt.get_cmap("rainbow")

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), dpi=300, constrained_layout=True)

    energy_ax = axs[0]
    force_ax = axs[1]
    energy_ax.plot(loss_hist['train_loss']['total']['energy'], label='train')
    energy_ax.plot(loss_hist['valid_loss']['total']['energy'], label='valid')
    energy_ax.set_xlabel('Epoch')
    energy_ax.set_ylabel('RMSE(eV/atom)')
    energy_ax.set_title('Energy')
    energy_ax.legend()

    force_ax.plot(loss_hist['train_loss']['total']['force'], label='train')
    force_ax.plot(loss_hist['valid_loss']['total']['force'], label='valid')
    force_ax.set_xlabel('Epoch')
    force_ax.set_ylabel(r'RMSE(eV/${\rm \AA}$)')
    force_ax.set_title('Force')
    force_ax.legend()

    plt.savefig(fname, format="png")
    """
