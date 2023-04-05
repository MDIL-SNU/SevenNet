import matplotlib
import matplotlib.pyplot as plt

from sevenn.train.trainer import DataSetType, LossType

matplotlib.use('pdf')


def draw_learning_curve(train_loss, valid_loss, fname):
    try:
        draw_learning_curve.history[DataSetType.TRAIN].append(train_loss)
        draw_learning_curve.history[DataSetType.VALID].append(valid_loss)
    except AttributeError:
        draw_learning_curve.history = {'train': [], 'valid': []}

    loss_hist = draw_learning_curve.history[5:]  # skip early loss(for visual)
    # do nothinig untill 10 epoch
    if len(loss_hist) < 10:
        return
    plt.clf()

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), dpi=300, constrained_layout=True)

    energy_ax = axs[0]
    force_ax = axs[1]
    energy_ax.plot(loss_hist['train']['total']['energy'], label='train')
    energy_ax.plot(loss_hist['valid']['total']['energy'], label='valid')
    energy_ax.set_xlabel('Epoch')
    energy_ax.set_ylabel('RMSE(eV/atom)')
    energy_ax.set_title('Energy')
    energy_ax.legend()

    force_ax.plot(loss_hist['train']['total']['force'], label='train')
    force_ax.plot(loss_hist['valid']['total']['force'], label='valid')
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
