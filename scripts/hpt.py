import os
from collections import defaultdict

import numpy as np
import pandas as pd
from skopt import Optimizer

from scripts.grafana import load_metrics


def load_past_results(
    name: str, parameters: list, loss_key: str = "valid_CELoss"
):
    hpt_dir = os.path.join("hpt", name)
    log_dir = os.path.join("logs", name)

    # First open and update the existing configurations.csv
    if os.path.exists(os.path.join(hpt_dir, "configurations.csv")):
        config_df = pd.read_csv(
            os.path.join(hpt_dir, "configurations.csv"), index_col=0
        )
        for i, log_run_dir in zip(config_df.index, config_df.log_dir):
            csv_path = os.path.join(log_run_dir, "csv_logs", "metrics.csv")
            if not os.path.exists(csv_path):
                print(f"Could not find {csv_path}")
                continue

            loss_df = load_metrics(
                csv_path,
                [loss_key],
            )[loss_key]
            min_loss = loss_df[loss_key].min()
            config_df.loc[i, loss_key] = min_loss

    else:
        config_df = pd.DataFrame(columns=parameters + [loss_key, "log_dir"])

    old_len = len(config_df)
    # Also add any configs with results not in configurations.csv
    to_add = defaultdict(list)
    for config in os.listdir(log_dir):
        if os.path.join(log_dir, config) in config_df["log_dir"].values:
            continue

        csv_path = os.path.join(log_dir, config, "csv_logs", "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"Could not find {csv_path}")
            continue

        param_values = config.split("+")
        param_values = {
            pv.split("@")[0]: pv.split("@")[1] for pv in param_values
        }
        expected_parameters = set(config_df.columns) - {"log_dir", loss_key}
        if set(param_values.keys()) != expected_parameters:
            print(
                f"Directory {os.path.join(log_dir, config)} does not have expected parameters {expected_parameters}"
            )

        loss_df = load_metrics(
            csv_path,
            [loss_key],
        )[loss_key]
        min_loss = loss_df[loss_key].min()

        for k, v in param_values.items():
            to_add[k].append(v)
        to_add[loss_key].append(min_loss)
        to_add["log_dir"].append(os.path.join(log_dir, config))

    config_df = pd.concat([config_df, pd.DataFrame(to_add)], ignore_index=True)
    print(
        f"Added {len(config_df) - old_len} configurations found in the logs directory"
    )

    config_df.to_csv(os.path.join(hpt_dir, "configurations.csv"))
    return config_df


def hpt(
    name: str,
    parameter_ranges: dict,
    n_initial_points: int,
    n_ask_points: int,
    loss_key: str = "valid_CELoss",
):
    hpt_dir = os.path.join("hpt", name)
    os.makedirs(hpt_dir, exist_ok=True)
    parameter_ranges = dict(sorted(parameter_ranges.items()))

    # Get HPT from name and load results
    config_df = load_past_results(name, list(parameter_ranges.keys()))

    # Create optimizer and tell
    optimizer = Optimizer(
        [v for v in parameter_ranges.values()],
        n_initial_points=n_initial_points,
    )

    for seen_config in config_df[config_df[loss_key].notnull()].itertuples():
        parameters = [getattr(seen_config, p) for p in parameter_ranges.keys()]
        optimizer.tell(parameters, getattr(seen_config, loss_key))

    new_configs = optimizer.ask(n_points=n_ask_points)
    new_configs = [
        [round(p, 8) if isinstance(p, (float, np.floating)) else p for p in c]
        for c in new_configs
    ]

    to_add = {
        p: [c[i] for c in new_configs]
        for i, p in enumerate(parameter_ranges.keys())
    } | {
        "log_dir": [
            os.path.join(
                "logs",
                name,
                "+".join(
                    [
                        f"{p}@{c[i]}"
                        for i, p in enumerate(parameter_ranges.keys())
                    ]
                ),
            )
            for c in new_configs
        ]
    }
    print(f"Added {len(to_add["log_dir"])} new configurations")
    config_df = pd.concat([config_df, pd.DataFrame(to_add)], ignore_index=True)
    config_df.to_csv(os.path.join(hpt_dir, "configurations.csv"))


if __name__ == "__main__":
    hpt(
        name="lr_scheduler",
        parameter_ranges={
            "learning_rate": (1e-6, 1e-2),
            "pct_start": (0, 0.9),
        },
        n_initial_points=5,
        n_ask_points=5,
    )
