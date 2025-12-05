from dataclasses import dataclass


@dataclass
class Metrics:
    precision: float = 0.0
    recall: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
