from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PrecisionCalibration:
    score_thresholds: list[float]
    precisions: list[float]
    iou_threshold: float
    total_detections: int

    def calibrate(self, score: float) -> float:
        for threshold, precision in zip(self.score_thresholds, self.precisions):
            if score >= threshold:
                return precision
        return 0.0

    def to_dict(self) -> dict:
        return {
            "score_thresholds": self.score_thresholds,
            "precisions": self.precisions,
            "iou_threshold": self.iou_threshold,
            "total_detections": self.total_detections,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrecisionCalibration":
        return cls(
            score_thresholds=[float(value) for value in data["score_thresholds"]],
            precisions=[float(value) for value in data["precisions"]],
            iou_threshold=float(data["iou_threshold"]),
            total_detections=int(data["total_detections"]),
        )


def fit_precision_calibration(
    detections: list[tuple[float, bool]],
    iou_threshold: float,
) -> PrecisionCalibration:
    if not detections:
        return PrecisionCalibration(score_thresholds=[], precisions=[], iou_threshold=iou_threshold, total_detections=0)

    sorted_detections = sorted(detections, key=lambda item: item[0], reverse=True)
    score_thresholds: list[float] = []
    precisions: list[float] = []
    true_positive_count = 0
    detections_seen = 0
    current_score: float | None = None

    for index, (score, is_true_positive) in enumerate(sorted_detections, start=1):
        if is_true_positive:
            true_positive_count += 1
        detections_seen = index
        next_score = sorted_detections[index][0] if index < len(sorted_detections) else None
        if current_score is None:
            current_score = score
        if next_score is None or next_score != score:
            score_thresholds.append(float(score))
            precisions.append(float(true_positive_count / detections_seen))

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    return PrecisionCalibration(
        score_thresholds=score_thresholds,
        precisions=precisions,
        iou_threshold=iou_threshold,
        total_detections=len(sorted_detections),
    )
