import os
from collections import defaultdict

import numpy as np
import pandas as pd
from skopt import Optimizer
from skopt.space import Integer, Real

from scripts.grafana import load_metrics


def build_search_space(parameter_ranges: dict):
    search_space = []
    exponent_map = {}  # param_name -> (index_in_space, base)

    for i, (name, spec) in enumerate(parameter_ranges.items()):
        # Special handling for exponent parameters
        if isinstance(spec, dict) and spec.get("type") == "exponent":
            base = spec.get("base", 2)
            low, high = spec["range"]
            search_space.append(Integer(low, high, name=name + "_exp"))
            exponent_map[name] = (i, base)

        # Anything else: pass through directly
        else:
            search_space.append(spec)

    return search_space, exponent_map


def decode_config(config, parameter_ranges, exponent_map):
    """Map optimizer values back into actual hyperparameter configs."""
    decoded = {}
    for i, (name, spec) in enumerate(parameter_ranges.items()):
        if name in exponent_map:
            _, base = exponent_map[name]
            decoded[name] = base ** config[i]
        else:
            decoded[name] = config[i]
    return decoded


def encode_config(config_dict, parameter_ranges, exponent_map):
    """Convert actual values back to optimizer space (for tell)."""
    encoded = []
    for name, spec in parameter_ranges.items():
        if name in exponent_map:
            _, base = exponent_map[name]
            # Find exponent that matches the actual value
            exp = int(round(np.log(config_dict[name]) / np.log(base)))
            encoded.append(exp)
        else:
            encoded.append(config_dict[name])
    return encoded


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

    if os.path.isdir(log_dir):
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
            expected_parameters = set(config_df.columns) - {
                "log_dir",
                loss_key,
            }
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

        config_df = pd.concat(
            [config_df, pd.DataFrame(to_add)], ignore_index=True
        )
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

    search_space, exponent_map = build_search_space(parameter_ranges)

    # Create optimizer and tell
    optimizer = Optimizer(
        search_space,
        n_initial_points=n_initial_points,
    )

    for seen_config in config_df[config_df[loss_key].notnull()].itertuples():
        params_dict = {
            p: getattr(seen_config, p) for p in parameter_ranges.keys()
        }
        optimizer.tell(
            encode_config(params_dict, parameter_ranges, exponent_map),
            getattr(seen_config, loss_key),
        )

    new_configs = optimizer.ask(n_points=n_ask_points)
    new_configs = [
        decode_config(c, parameter_ranges, exponent_map) for c in new_configs
    ]
    new_configs = [
        {
            p: round(v, 8) if isinstance(v, (float, np.floating)) else v
            for p, v in c.items()
        }
        for c in new_configs
    ]

    to_add = {
        p: [c[p] for c in new_configs] for p in parameter_ranges.keys()
    } | {
        "log_dir": [
            os.path.join(
                "logs",
                name,
                "+".join([f"{p}@{c[p]}" for p in parameter_ranges.keys()]),
            )
            for c in new_configs
        ]
    }
    print(f"Added {len(to_add["log_dir"])} new configurations")
    config_df = pd.concat([config_df, pd.DataFrame(to_add)], ignore_index=True)
    config_df.to_csv(os.path.join(hpt_dir, "configurations.csv"))


if __name__ == "__main__":
    # hpt(
    #     name="lr_scheduler",
    #     parameter_ranges={
    #         "learning_rate": (1e-6, 1e-2),
    #         "pct_start": (0, 0.9),
    #     },
    #     n_initial_points=5,
    #     n_ask_points=5,
    # )
    hpt(
        name="bs_lr_default",
        parameter_ranges={
            "learning_rate": Real(1e-5, 5e-3, prior="log-uniform"),
            "global_train_batch_size": {
                "type": "exponent",
                "base": 2,
                "range": (4, 9),
            },
        },
        n_initial_points=15,
        n_ask_points=15,
    )
