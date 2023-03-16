import matplotlib
import matplotlib.pyplot as plt

from sevenn.train.trainer import DataSetType

matplotlib.use('pdf')


def draw_learning_curve(loss_hist, fname):
    # do nothinig untill 10 epoch
    if len(loss_hist[DataSetType.TRAIN]['total']['energy']) < 10:
        return
    plt.clf()

    # cut unusually large loss at early epoch for visual
    # TODO: refactor
    for i in range(5):
        t_E = loss_hist[DataSetType.TRAIN]['total']['energy'][i]
        t_F = loss_hist[DataSetType.TRAIN]['total']['force'][i]
        v_E = loss_hist[DataSetType.VALID]['total']['energy'][i]
        v_F = loss_hist[DataSetType.VALID]['total']['force'][i]
        if t_E > 1.0 or t_F > 1.0 or v_E > 1.0 or v_F > 1.0:
            loss_hist[DataSetType.TRAIN]['total']['energy'] = \
                loss_hist[DataSetType.TRAIN]['total']['energy'][i:]
            loss_hist[DataSetType.TRAIN]['total']['force'] = \
                loss_hist[DataSetType.TRAIN]['total']['force'][i:]
            loss_hist[DataSetType.VALID]['total']['energy'] = \
                loss_hist[DataSetType.VALID]['total']['energy'][i:]
            loss_hist[DataSetType.VALID]['total']['force'] = \
                loss_hist[DataSetType.VALID]['total']['force'][i:]

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
