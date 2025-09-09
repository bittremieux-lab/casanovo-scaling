import copy
import itertools
import os
import subprocess
from datetime import datetime

import pandas as pd
import yaml


def get_default_config(experiment):
    with open(
        os.path.join("hpc_scripts", experiment, "default.yaml"), "r"
    ) as d_conf:
        return yaml.safe_load(d_conf)


def create_config(experiment, default_config, **kwargs):
    config = copy.deepcopy(default_config)
    config.update(kwargs)

    parameter_str = "+".join([f"{k}@{v}" for k, v in kwargs.items()])
    new_config_path = os.path.join(
        "hpc_scripts", experiment, f"{parameter_str}.yaml"
    )
    new_bs_config_path = os.path.join(
        "hpc_scripts", experiment, f"{parameter_str}__bs.yaml"
    )

    with open(new_config_path, "w") as new_conf:
        yaml.dump(config, new_conf)
    return new_config_path, new_bs_config_path, parameter_str


def submit_job(
    experiment,
    train_file,
    val_file,
    param_combination,
    default_config,
    echo_only,
):
    new_config_path, new_bs_config_path, parameter_str = create_config(
        experiment, default_config, **param_combination
    )

    output_dir = os.path.join("logs", experiment, parameter_str)
    pbs_log_file = os.path.join(
        output_dir, f"pbs_{datetime.now().strftime('%Y%m%d-%H%M%S')}.out"
    )

    env_vars = {
        "TRAIN_FILE": train_file,
        "VAL_FILE": val_file,
        "OUTPUT_DIR": output_dir,
        "CONFIG_FILE": new_config_path,
        "BS_CONFIG_FILE": new_bs_config_path,
    }

    pbs_file = os.path.join("hpc_scripts", experiment, f"{experiment}.pbs")

    command = [
        "qsub",
        "-o",
        pbs_log_file,
        "-j",
        "oe",
        pbs_file,
        "-v",
        ",".join(f"{k}={v}" for k, v in env_vars.items()),
    ]

    subprocess.run(["echo"] + command, check=True)
    if not echo_only:
        subprocess.run(command, check=True)


def submit_grid_commands(
    experiment, train_file, val_file, exclude=None, echo_only=False, **kwargs
):
    default_config = get_default_config(experiment)
    kwargs = dict(sorted(kwargs.items()))

    for param_combination in itertools.product(*kwargs.values()):
        param_combination = dict(zip(kwargs.keys(), param_combination))

        if exclude is not None and param_combination in exclude:
            continue

        submit_job(
            experiment,
            train_file,
            val_file,
            param_combination,
            default_config,
            echo_only,
        )


def submit_hpt_commands(
    experiment, train_file, val_file, hpt_ids, echo_only=False
):
    default_config = get_default_config(experiment)
    hpt_file = os.path.join("hpt", experiment, f"configurations.csv")
    hpt_df = pd.read_csv(hpt_file, index_col=0)

    for hpt_id in hpt_ids:
        param_combination = hpt_df.loc[hpt_id]
        param_combination = param_combination.to_dict()
        param_combination.pop("log_dir")
        param_combination.pop("valid_CELoss")

        submit_job(
            experiment,
            train_file,
            val_file,
            param_combination,
            default_config,
            echo_only,
        )


if __name__ == "__main__":
    train_file = "massivekb_data/scaling_data_max_100000/train_2s_1000000p.mgf"
    val_file = "massivekb_data/scaling_data_max_100000/val_0.25.mgf"

    # submit_grid_commands(
    #     experiment="old_optim_scheduler",
    #     train_file=train_file,
    #     val_file=val_file,
    #     learning_rate=[1e-4, 1.6e-4, 2.5e-4, 4e-4, 6.3e-4, 1e-3],
    # )
    # submit_hpt_commands(
    #     experiment="lr_scheduler",
    #     train_file=train_file,
    #     val_file=val_file,
    #     hpt_ids=range(21, 26),
    # )
    submit_hpt_commands(
        experiment="bs_lr_default",
        train_file=train_file,
        val_file=val_file,
        hpt_ids=range(20, 25),
    )
