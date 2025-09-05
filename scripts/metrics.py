import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def metrics_from_hpt(experiment, x, y, z, max_z=None):
    hpt_config = pd.read_csv(
        os.path.join("hpt", experiment, "configurations.csv")
    )
    if max_z is not None:
        hpt_config[z] = hpt_config[z].clip(upper=max_z)
    return hpt_config[x], hpt_config[y], hpt_config[z]


def plot_training_metrics(csv_path):
    """
    Reads a CSV with columns:
    epoch, hp/optimizer_cosine_schedule_period_iters, hp/optimizer_warmup_iters,
    lr-Adam, lr-Adam-momentum, lr-Adam-weight_decay, step, train_CELoss, valid_CELoss
    and creates three plots with step on the x-axis.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Ensure step is sorted in case the CSV isn't
    df = df.sort_values(by="step")

    print(df)
    print(df["train_CELoss"])

    # 1. Plot learning rate
    mask = df["lr-Adam"].notna()
    plt.figure(figsize=(8, 4))
    plt.plot(
        df.loc[mask, "step"],
        df.loc[mask, "lr-Adam"],
        label="Learning Rate (Adam)",
    )
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs Step")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Train & Validation Loss ---
    plt.figure(figsize=(8, 4))

    # Train CE Loss
    mask_train = df["train_CELoss"].notna()
    plt.plot(
        df.loc[mask_train, "step"],
        df.loc[mask_train, "train_CELoss"],
        label="Train CE Loss",
        color="orange",
    )

    # Validation CE Loss
    mask_valid = df["valid_CELoss"].notna()
    plt.plot(
        df.loc[mask_valid, "step"],
        df.loc[mask_valid, "valid_CELoss"],
        label="Valid CE Loss",
        color="red",
    )
    plt.xlabel("Step")
    plt.ylabel("CE Loss")
    plt.title("Train & Validation CE Loss vs Step")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_train_subsets_grid(root_dir="logs/casanovo_train_subsets"):
    heatmap = np.full((5, 5), np.nan)

    spectra = []
    peptides = []
    vals = {}
    for d in os.listdir(root_dir):
        spec, pep = d.split("_")
        spec = int(spec[:-1])
        pep = int(pep[:-1])
        spectra.append(spec)
        peptides.append(pep)
        metrics_df = pd.read_csv(
            os.path.join(root_dir, d, "csv_logs", "metrics.csv")
        )
        val = metrics_df["valid_CELoss"].min()
        if val > 0.6:
            val = np.nan
        vals[pep, spec] = val

    spectra = sorted(set(spectra))
    peptides = sorted(set(peptides))
    for s in spectra:
        for p in peptides:
            val = vals[p, s]
            heatmap[peptides.index(p), spectra.index(s)] = val
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
    )

    plt.colorbar(im, label="Min validation loss")
    plt.xticks(ticks=range(len(spectra)), labels=spectra)
    plt.yticks(ticks=range(len(peptides)), labels=peptides)
    plt.xlabel("# Spectra")
    plt.ylabel("# Unique peptides")
    plt.tight_layout()
    plt.show()


def plot_2D_heatmap(experiment, x, y, z, from_hpt=True, max_z=None):
    x_v, y_v, z_v = metrics_from_hpt(experiment, x, y, z, max_z)

    # Transform to log space
    logx, logy = np.log10(x_v), np.log10(y_v)

    # Define margins in log space (10% of log-range)
    margin_logx = 0.1 * (logx.max() - logx.min())
    margin_logy = 0.1 * (logy.max() - logy.min())

    # Define log grid
    grid_logx, grid_logy = np.mgrid[
        (logx.min() - margin_logx) : (logx.max() + margin_logx) : 200j,
        (logy.min() - margin_logy) : (logy.max() + margin_logy) : 200j,
    ]

    # Interpolate in log space
    grid_z = griddata(
        (logx, logy), z_v, (grid_logx, grid_logy), method="cubic"
    )

    # Convert grid back to linear scale for plotting
    grid_X, grid_Y = 10**grid_logx, 10**grid_logy

    # Plot
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(
        grid_X, grid_Y, grid_z, shading="auto", cmap="viridis_r", alpha=0.7
    )

    # Scatter the actual data points
    plt.scatter(x_v, y_v, c=z_v, cmap="viridis_r", edgecolor="k", s=100)

    plt.colorbar(label=z)
    plt.xscale("log")
    plt.yscale("log", base=2)
    plt.xlabel(x)
    plt.ylabel(y)
    # plt.title("Interpolated heatmap in log-log space")
    plt.show()


if __name__ == "__main__":
    # plot_training_metrics(
    #     "logs/casanovo_train_subsets/1s_100000p/csv_logs/metrics.csv"
    # )
    # plot_train_subsets_grid()

    plot_2D_heatmap(
        "bs_lr_default",
        x="learning_rate",
        y="global_train_batch_size",
        z="valid_CELoss",
        max_z=1,
    )
