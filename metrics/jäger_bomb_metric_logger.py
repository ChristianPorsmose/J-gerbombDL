import csv

from engine.data import BatchResult
from pathlib import Path
from utils.echo import log


class JägerBombMetricLogger:
    """
    Metric logger for Jäger Bomb training.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._write_headers()

    def _write_headers(self):
        csv_headers = [
            "epoch",
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "train/spatial_loss",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
            "val/spatial_loss",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "lr/pg0",
        ]
        with open(self.file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    def log_batch_result(self, batchResult: BatchResult, epoch: int, lr: float):
        """
        Log epoch results to CSV and print to terminal.
        """
        metrics = batchResult.metrics
        log(f"METRICS — Epoch {epoch}: ")
        log(f"P: {metrics.precision:.4f}, ")
        log(f"R: {metrics.recall:.4f}, ")
        log(f"mAP50: {metrics.mAP50:.4f}, ")
        log(f"mAP50-95: {metrics.mAP50_95:.4f}")

        row = [
            epoch,
            batchResult.train_loss.box,
            batchResult.train_loss.cls,
            batchResult.train_loss.dfl,
            batchResult.train_loss.spatial,
            batchResult.val_loss.box,
            batchResult.val_loss.cls,
            batchResult.val_loss.dfl,
            batchResult.val_loss.spatial,
            batchResult.metrics.precision,
            batchResult.metrics.recall,
            batchResult.metrics.mAP50,
            batchResult.metrics.mAP50_95,
            lr,
        ]

        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
