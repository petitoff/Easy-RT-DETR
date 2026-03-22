import pytest

from easy_rtdetr.score_calibration import PrecisionCalibration, fit_precision_calibration


def test_fit_precision_calibration_builds_monotonic_mapping():
    calibration = fit_precision_calibration(
        detections=[
            (0.90, True),
            (0.80, False),
            (0.80, True),
            (0.50, False),
        ],
        iou_threshold=0.5,
    )

    assert calibration.score_thresholds == pytest.approx([0.9, 0.8, 0.5])
    assert calibration.precisions == pytest.approx([1.0, 2.0 / 3.0, 0.5])
    assert calibration.calibrate(0.95) == pytest.approx(1.0)
    assert calibration.calibrate(0.85) == pytest.approx(2.0 / 3.0)
    assert calibration.calibrate(0.10) == pytest.approx(0.0)


def test_precision_calibration_round_trip():
    calibration = PrecisionCalibration(
        score_thresholds=[0.9, 0.8],
        precisions=[1.0, 0.75],
        iou_threshold=0.5,
        total_detections=10,
    )
    restored = PrecisionCalibration.from_dict(calibration.to_dict())
    assert restored == calibration
