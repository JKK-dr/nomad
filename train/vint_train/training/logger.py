import numpy as np
import csv
import os
import time


def _to_float(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def append_metrics_to_csv(
    project_folder: str,
    phase: str,
    dataset: str,
    epoch: int,
    batch: int,
    metrics: dict,
    lr: float = None,
):
    os.makedirs(project_folder, exist_ok=True)
    csv_path = os.path.join(project_folder, "metrics.csv")
    write_header = not os.path.exists(csv_path)
    project_name = os.path.basename(os.path.dirname(project_folder))
    run_name = os.path.basename(project_folder)
    fieldnames = [
        "timestamp",
        "project_name",
        "run_name",
        "phase",
        "dataset",
        "epoch",
        "batch",
        "lr",
        "metric",
        "value",
    ]

    rows = []
    for metric, value in metrics.items():
        try:
            value = _to_float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(value):
            continue
        rows.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_name": project_name,
            "run_name": run_name,
            "phase": phase,
            "dataset": dataset,
            "epoch": epoch,
            "batch": batch,
            "lr": "" if lr is None else lr,
            "metric": metric,
            "value": value,
        })

    if len(rows) == 0:
        return

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


class Logger:
    def __init__(
        self,
        name: str,
        dataset: str,
        window_size: int = 10,
        rounding: int = 4,
    ):
        """
        Args:
            name (str): Name of the metric
            dataset (str): Name of the dataset
            window_size (int, optional): Size of the moving average window. Defaults to 10.
            rounding (int, optional): Number of decimals to round to. Defaults to 4.
        """
        self.data = []
        self.name = name
        self.dataset = dataset
        self.rounding = rounding
        self.window_size = window_size

    def display(self) -> str:
        latest = round(self.latest(), self.rounding)
        average = round(self.average(), self.rounding)
        moving_average = round(self.moving_average(), self.rounding)
        output = f"{self.full_name()}: {latest} ({self.window_size}pt moving_avg: {moving_average}) (avg: {average})"
        return output

    def log_data(self, data: float):
        if not np.isnan(data):
            self.data.append(data)

    def full_name(self) -> str:
        return f"{self.name} ({self.dataset})"

    def latest(self) -> float:
        if len(self.data) > 0:
            return self.data[-1]
        return np.nan

    def average(self) -> float:
        if len(self.data) > 0:
            return np.mean(self.data)
        return np.nan

    def moving_average(self) -> float:
        if len(self.data) > self.window_size:
            return np.mean(self.data[-self.window_size :])
        return self.average()
