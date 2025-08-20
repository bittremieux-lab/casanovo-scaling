from datetime import datetime

import yaml
import subprocess

TRAIN_FILE = "massivekb_data/scaling_data_max_100000/train_2s_1000000p.mgf"
VAL_FILE = "massivekb_data/scaling_data_max_100000/val.mgf"


def create_config(default_config, path, **kwargs):
    config = default_config.copy()
    config.update(kwargs)
    setting_str = "_".join([f"{v}{k}" for k, v in kwargs.items()])
    CONFIG_FILE = f"{path}_{setting_str}.yaml"

    with open(CONFIG_FILE, "w") as fw:
        yaml.dump(config, fw)


with open("hpc_scripts/lr_scheduler/lr_scheduler_default.yaml", "r") as f:
    default_config = yaml.safe_load(f)

lrs = [1e-6, 1e-5, 1e-4]
lrs = [2e-3]
pct_starts = [0.15]

for lr in lrs:
    for pct_start in pct_starts:
        create_config(
            default_config,
            "hpc_scripts/lr_scheduler/lr_scheduler",
            lr=lr,
            pct_start=pct_start,
        )

        OUTPUT_DIR = f"logs/lr_scheduler/{lr}lr_{pct_start}pct_start/"
        PBS_LOG_FILE = (
            f"{OUTPUT_DIR}pbs_{datetime.now().strftime('%Y%m%d-%H%M%S')}.out"
        )

        echo_command = [
            "echo",
            "qsub",
            "-o",
            PBS_LOG_FILE,
            "-j",
            "oe",
            "hpc_scripts/lr_scheduler/lr_scheduler.pbs",
            "-v",
            f"TRAIN_FILE={TRAIN_FILE},VAL_FILE={VAL_FILE},OUTPUT_DIR={OUTPUT_DIR}",
        ]
        subprocess.run(echo_command, check=True)

        # qsub_command = [
        #     "qsub",
        #     "-o",
        #     PBS_LOG_FILE,
        #     "-j",
        #     "oe",
        #     "hpc_scripts/lr_scheduler/lr_scheduler.pbs",
        #     "-v",
        #     f"TRAIN_FILE={TRAIN_FILE},VAL_FILE={VAL_FILE},OUTPUT_DIR={OUTPUT_DIR},CONFIG_FILE={CONFIG_FILE}",
        # ]
        # subprocess.run(qsub_command, check=True)
