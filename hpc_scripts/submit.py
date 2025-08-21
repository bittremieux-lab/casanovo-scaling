import copy
import itertools
import os
import subprocess
from datetime import datetime

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

    with open(new_config_path, "w") as new_conf:
        yaml.dump(config, new_conf)
    return new_config_path, parameter_str


def submit_commands(
    experiment, train_file, val_file, exclude=None, echo_only=False, **kwargs
):
    default_config = get_default_config(experiment)
    pbs_file = os.path.join("hpc_scripts", experiment, f"{experiment}.pbs")
    kwargs = dict(sorted(kwargs.items()))

    for param_combination in itertools.product(*kwargs.values()):
        param_combination = dict(zip(kwargs.keys(), param_combination))

        if exclude is not None and param_combination in exclude:
            continue

        new_config_path, parameter_str = create_config(
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
        }

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


if __name__ == "__main__":
    train_file = "massivekb_data/scaling_data_max_100000/train_2s_1000000p.mgf"
    val_file = "massivekb_data/scaling_data_max_100000/val_0.25.mgf"

    submit_commands(
        experiment="lr_scheduler",
        train_file=train_file,
        val_file=val_file,
        exclude=[
            {"learning_rate": 1e-4, "pct_start": 0.3},
        ],
        learning_rate=[1e-4, 3e-4, 1e-3],
        pct_start=[0.15, 0.3],
    )
