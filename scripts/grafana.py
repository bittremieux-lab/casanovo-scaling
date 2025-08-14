import datetime
import hashlib
import json
import os

import pandas as pd
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

STATE_FILE = "logs/metrics_state.json"


def setup_db(
    token="2mBaDlegqP4bIeal1s6EXm5ciwWh-hfeboKHwulUIm7peWti49_PCiab-K7hkdYBmzyenOw0b0F0RckzvskDDg==",
    org="casanovo-scaling",
):
    url = "http://localhost:8086"

    client = InfluxDBClient(url=url, token=token, org=org)
    return client


def file_hash(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def load_metrics(csv_path, keys):
    df = pd.read_csv(csv_path)
    dfs = {}
    for key in keys:
        dfs[key] = df[df[key].notna()]
    return dfs


def sync_metrics(dbclient, log_dir="logs/casanovo_train_subsets/"):
    write_api = dbclient.write_api(write_options=SYNCHRONOUS)
    state = load_state()
    updated_state = {}

    for run in os.listdir(log_dir):
        csv_path = os.path.join(log_dir, run, "csv_logs", "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"Could not find {csv_path}")
            continue

        current_hash = file_hash(csv_path)
        if state.get(run) == current_hash:
            # Skip unchanged files
            print(f"{csv_path} unchanged, skipping")
            continue

        print(f"Processing {csv_path}")
        metrics_dfs = load_metrics(
            csv_path,
            ["lr-Adam", "train_CELoss", "valid_CELoss"],
        )
        points = []
        for metric_type, metric_df in metrics_dfs.items():
            for _, row in metric_df.iterrows():
                start_time = datetime.datetime(2025, 8, 1)
                point_time = start_time + datetime.timedelta(
                    seconds=int(row.step)
                )
                points.append(
                    Point("step_metrics")
                    .tag("run", run)
                    .tag("type", metric_type)
                    .field("step", row.step)
                    .field("value", row[metric_type])
                    .time(point_time)
                )
        if points:
            write_api.write(
                bucket="train_subsets", org="casanovo-scaling", record=points
            )

        updated_state[run] = current_hash

    save_state({**state, **updated_state})


if __name__ == "__main__":
    dbclient = setup_db()
    sync_metrics(dbclient)
