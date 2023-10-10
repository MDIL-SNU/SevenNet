import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

from sevenn._const import DataSetType, LossType

matplotlib.use('pdf')


def draw_learning_curve(loss_history, fname):
    # alias
    CUT = 5
    train_hist = loss_history[DataSetType.TRAIN]
    valid_hist = loss_history[DataSetType.VALID]
    epoch_num = len(train_hist[LossType.ENERGY])

    # do nothinig untill 10 epoch
    if epoch_num < 10:
        return
    plt.clf()

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), dpi=300, constrained_layout=True)
    x = list(range(CUT, epoch_num))

    energy_ax = axs[0]
    force_ax = axs[1]
    energy_ax.plot(x, train_hist[LossType.ENERGY][5:], label='train')
    energy_ax.plot(x, valid_hist[LossType.ENERGY][5:], label='valid')
    #energy_ax.set_xlim(3, epoch_num - 1)
    energy_ax.set_xlabel('Epoch')
    energy_ax.set_ylabel('RMSE (eV/atom)')
    energy_ax.set_title('Energy')
    energy_ax.legend()

    force_ax.plot(x, train_hist[LossType.FORCE][5:], label='train')
    force_ax.plot(x, valid_hist[LossType.FORCE][5:], label='valid')
    #force_ax.set_xlim(3, epoch_num - 1)
    force_ax.set_xlabel('Epoch')
    force_ax.set_ylabel(r'RMSE (eV/${\rm \AA}$)')
    force_ax.set_title('Force')
    force_ax.legend()

    plt.savefig(fname, format="png")
    plt.close()


# TODO: refactoring
def _draw_parity(parity, rmses, fname):
    # implement atom wise, label wise parity plots if you want..
    PARAMS = [('E$_{DFT}$ (eV/atom)', 'E$_{NNP}$ (eV/atom)', 0.2),
              (r'F$_{DFT}$ (eV/${\rm \AA}$)', r'F$_{NNP}$ (eV/${\rm \AA}$)', 5),
              ('S$_{DFT}$ (kB)', 'S$_{NNP}$ (kB)', 10)]
    MAX_SAMPLE = 10000

    is_stress = len(parity["stress"]["ref"]) != 0
    rmses = rmses.values()

    ACTUAL_DATA_KEYS = ["energy", "force"]
    if is_stress:
        ACTUAL_DATA_KEYS.append("stress")
        # ignore shear stress
        parity['stress']['pred'] = \
            [v for i, v in enumerate(parity['stress']['pred']) if i % 6 < 3]
        parity['stress']['ref'] = \
            [v for i, v in enumerate(parity['stress']['ref']) if i % 6 < 3]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    parity = {k: parity[k] for k in ACTUAL_DATA_KEYS}

    # Create a color map for heatmap
    cmap = plt.get_cmap('coolwarm')

    # Maximum number of samples to use for KDE

    # Iterate over your dictionary items
    for ax, (key, value_dict), PARAM, rmse in zip(axs, parity.items(), PARAMS, rmses):
        x_lb, y_lb, bw = PARAM

        ref = value_dict['ref']
        pred = value_dict['pred']
        # If there are more than max_samples, take a random subset
        if len(value_dict['pred']) > MAX_SAMPLE:
            idx = np.random.choice(len(pred), MAX_SAMPLE, replace=False)
            kde_pred = np.array(pred)[idx]
            kde_ref = np.array(ref)[idx]
        else:
            kde_pred = pred
            kde_ref = ref

        # Get min and max values across both pred and ref for this key
        min_val = min(min(pred), min(ref))
        max_val = max(max(pred), max(ref))

        # Compute the Kernel Density Estimate
        xy = np.vstack([kde_ref, kde_pred]).T
        kde = KernelDensity(bandwidth=bw, metric='euclidean',
                            kernel='gaussian', algorithm='ball_tree')
        kde.fit(xy)

        xy_all = np.vstack([ref, pred]).T
        z = np.exp(kde.score_samples(xy_all))

        # Scatter plot colored by density
        scatter = ax.scatter(ref, pred, c=z, s=50, edgecolors='none', cmap=cmap)

        # Plot diagonal
        ax.plot([min_val, max_val], [min_val, max_val],
                color='black', linestyle='--')

        ax.text(0.05, 0.95, f"RMSE: {rmse}",
                transform=ax.transAxes, ha='left', va='top')

        # Square aspect ratio and same scale
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])

        # Set labels
        ax.set_xlabel(x_lb)
        ax.set_ylabel(y_lb)
        ax.set_title(key.capitalize())

        # Create a colorbar for this subplot
        fig.colorbar(scatter, ax=ax, orientation='vertical')

    # Show plot
    plt.savefig(fname, format="png")
    plt.close()


def draw_every_parity(train_parity, valid_parity, train_loss, valid_loss, prefix):
    # To single image file would be better + seperate density scatter plot from func
    _draw_parity(train_parity, train_loss['total'], f"{prefix}_train.png")
    _draw_parity(valid_parity, valid_loss['total'], f"{prefix}_valid.png")


if __name__ == "__main__":
    import torch
    fn = torch.load("./parity_at_300.pth", map_location=torch.device("cpu"))
    draw_every_parity(fn['train'], fn['valid'], "./")
