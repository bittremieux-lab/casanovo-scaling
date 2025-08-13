import pandas as pd
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    plot_training_metrics(
        "logs/casanovo_train_subsets/1s_100000p/csv_logs/metrics.csv"
    )
