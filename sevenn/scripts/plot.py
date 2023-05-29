import matplotlib
import matplotlib.pyplot as plt

from sevenn.train.trainer import DataSetType, LossType

matplotlib.use('pdf')


def draw_learning_curve(loss_history, fname):
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
    plt.close()


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
