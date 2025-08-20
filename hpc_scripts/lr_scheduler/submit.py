from datetime import datetime

import yaml
import subprocess

TRAIN_FILE = "massivekb_data/scaling_data_max_100000/train_2s_1000000p.mgf"
VAL_FILE = "massivekb_data/scaling_data_max_100000/val.mgf"

with open("hpc_scripts/lr_scheduler/lr_scheduler_default.yaml", "r") as f:
    default_config = yaml.safe_load(f)

    lrs = [1e-6, 1e-5, 1e-4]
    for lr in lrs:
        config = default_config.copy()
        config["learning_rate"] = lr

        OUTPUT_DIR = f"logs/lr_scheduler/lr_{lr}/"
        PBS_LOG_FILE = (
            f"{OUTPUT_DIR}pbs_{datetime.now().strftime('%Y%m%d-%H%M%S')}.out"
        )
        CONFIG_FILE = f"hpc_scripts/lr_scheduler/lr_scheduler_{lr}.yaml"

        with open(CONFIG_FILE, "w") as fw:
            yaml.dump(config, fw)

        echo_command = [
            "echo",
            "-o",
            PBS_LOG_FILE,
            "-j",
            "oe",
            "hpc_scripts/lr_scheduler/lr_scheduler.pbs",
            "-v",
            f"TRAIN_FILE={TRAIN_FILE},VAL_FILE={VAL_FILE},OUTPUT_DIR={OUTPUT_DIR}",
        ]
        result = subprocess.run(echo_command, capture_output=True)
        print(result.stdout)
        print(result.stderr)

        qsub_command = [
            "qsub",
            "-o",
            PBS_LOG_FILE,
            "-j",
            "oe",
            "hpc_scripts/lr_scheduler/lr_scheduler.pbs",
            "-v",
            f"TRAIN_FILE={TRAIN_FILE},VAL_FILE={VAL_FILE},OUTPUT_DIR={OUTPUT_DIR},CONFIG_FILE={CONFIG_FILE}",
        ]
        result = subprocess.run(qsub_command, capture_output=True)
        print(result.stdout)
        print(result.stderr)
        print()
