import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import griddata

from scripts.hpt import load_past_results


def metrics_from_hpt(experiment, x, y, z, max_z=None):
    hpt_config = pd.read_csv(
        os.path.join("hpt", experiment, "configurations.csv")
    )
    hpt_config = hpt_config[hpt_config[z].notna()]
    if max_z is not None:
        hpt_config[z] = hpt_config[z].clip(upper=max_z)
    return hpt_config[x], hpt_config[y], hpt_config[z]


def parse_dir_name(name: str) -> dict:
    """
    Parse directory names of the form:
    key1@val1+key2@val2+...+keyn@valn

    Returns a dict {key1: val1, key2: val2, ...}.
    """
    params = {}
    parts = name.split("+")
    for part in parts:
        if "@" not in part:
            continue
        key, val = part.split("@", 1)
        # try to cast values to float if possible
        try:
            val = float(val)
        except ValueError:
            pass
        params[key] = val
    return params


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


def plot_lr_scheduler_results(root_dir="logs/introducing_new"):
    schedulers = {}

    for d in os.listdir(root_dir):
        params = parse_dir_name(d)
        if "learning_rate" not in params or "lr_scheduler" not in params:
            continue

        lr = params["learning_rate"]
        scheduler = params["lr_scheduler"]

        # Special case: onecycle with cycle_momentum=False
        if (
            scheduler == "onecycle"
            and str(params.get("cycle_momentum", "True")).lower() == "false"
        ):
            scheduler_label = "onecycle (no momentum)"
        else:
            scheduler_label = scheduler

        metrics_path = os.path.join(root_dir, d, "csv_logs", "metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        metrics_df = pd.read_csv(metrics_path)
        if "valid_CELoss" not in metrics_df.columns:
            continue

        val = metrics_df["valid_CELoss"].min()
        # if val > 0.6:
        #     val = np.nan

        if scheduler_label not in schedulers:
            schedulers[scheduler_label] = {"lr": [], "loss": []}

        schedulers[scheduler_label]["lr"].append(lr)
        schedulers[scheduler_label]["loss"].append(val)

    # Plot results
    plt.figure(figsize=(7, 5))
    for scheduler, data in schedulers.items():
        lrs = np.array(data["lr"])
        losses = np.array(data["loss"])

        # Sort by learning rate for nicer plotting
        order = np.argsort(lrs)
        lrs = lrs[order]
        losses = losses[order]

        (line,) = plt.plot(lrs, losses, marker="o", label=scheduler)
        line_color = line.get_color()

        # Highlight best (lowest loss) point with open circle
        if np.isfinite(losses).any():
            best_idx = np.nanargmin(losses)
            plt.scatter(
                lrs[best_idx],
                losses[best_idx],
                facecolors="none",
                edgecolors=line_color,
                s=200,
                linewidths=2,
                zorder=5,
            )

    plt.xscale("log")  # log scale for LR usually makes sense
    plt.xlabel("Learning rate")
    plt.ylabel("Min validation loss")
    plt.title("Validation loss vs learning rate per scheduler")
    plt.legend(title="LR Scheduler")
    plt.tight_layout()
    plt.show()


def plot_gradient_clip_results(root_dir="logs/introducing_new"):
    clip_groups = {}

    for d in os.listdir(root_dir):
        params = parse_dir_name(d)
        if "learning_rate" not in params or "gradient_clip_val" not in params:
            continue

        lr = params["learning_rate"]
        clip_val = params["gradient_clip_val"]

        metrics_path = os.path.join(root_dir, d, "csv_logs", "metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        metrics_df = pd.read_csv(metrics_path)
        if "valid_CELoss" not in metrics_df.columns:
            continue

        val = metrics_df["valid_CELoss"].min()
        if val > 0.6:  # filter like before
            val = np.nan

        if clip_val not in clip_groups:
            clip_groups[clip_val] = {"lr": [], "loss": []}

        clip_groups[clip_val]["lr"].append(lr)
        clip_groups[clip_val]["loss"].append(val)

    # Plot results
    plt.figure(figsize=(7, 5))
    for clip_val, data in clip_groups.items():
        lrs = np.array(data["lr"])
        losses = np.array(data["loss"])

        # Sort by learning rate for nicer plotting
        order = np.argsort(lrs)
        lrs = lrs[order]
        losses = losses[order]

        # Plot line and get its color
        (line,) = plt.plot(lrs, losses, marker="o", label=f"clip={clip_val:g}")
        line_color = line.get_color()

        # Highlight best (lowest loss) point with open circle in same color
        if np.isfinite(losses).any():
            best_idx = np.nanargmin(losses)
            plt.scatter(
                lrs[best_idx],
                losses[best_idx],
                facecolors="none",
                edgecolors=line_color,
                s=200,
                linewidths=2,
                zorder=5,
            )

    plt.xscale("log")  # log scale for LR
    plt.xlabel("Learning rate")
    plt.ylabel("Min validation loss")
    plt.title("Validation loss vs learning rate per gradient clip value")
    plt.legend(title="Gradient Clip", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_2D_heatmap(experiment, x, y, z, max_z=None, method="nearest"):
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
    grid_z = griddata((logx, logy), z_v, (grid_logx, grid_logy), method=method)

    # Convert grid back to linear scale for plotting
    grid_X, grid_Y = 10**grid_logx, 10**grid_logy

    # Plot
    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap("viridis_r")
    norm = colors.Normalize(vmin=z_v.min(), vmax=z_v.max())

    plt.pcolormesh(grid_X, grid_Y, grid_z, shading="auto", cmap=cmap)

    min_idx = np.argmin(z_v)
    mask = np.ones_like(x_v, dtype=bool)
    mask[min_idx] = False

    # Scatter the actual data points
    scatter = plt.scatter(
        x_v[mask], y_v[mask], c=z_v[mask], cmap=cmap, edgecolor="k", s=100
    )

    # Highlight the minimum
    plt.scatter(
        x_v[min_idx],
        y_v[min_idx],
        c=[cmap(norm(z_v[min_idx]))],
        edgecolor="k",
        s=300,
        marker="*",
    )

    plt.colorbar(scatter, label=z)
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

    # load_past_results(
    #     name="bs_lr_default",
    #     parameters=["learning_rate", "global_train_batch_size"],
    #     loss_key="valid_CELoss",
    # )
    #
    # plot_2D_heatmap(
    #     "bs_lr_default",
    #     x="learning_rate",
    #     y="global_train_batch_size",
    #     z="valid_CELoss",
    #     max_z=1,
    #     method="nearest",
    # )
    # plot_2D_heatmap(
    #     "bs_lr_default",
    #     x="learning_rate",
    #     y="global_train_batch_size",
    #     z="valid_CELoss",
    #     max_z=1,
    #     method="cubic",
    # )
    # plot_lr_scheduler_results()
    plot_gradient_clip_results()
