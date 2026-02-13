from datetime import date, timedelta

from app.models.metrics import acwr, efficiency_index, detect_gps_drift, detect_hr_anomalies


def test_acwr_calculation_accuracy() -> None:
    start = date(2026, 1, 1)
    series = [(start + timedelta(days=i), 100.0) for i in range(28)]
    output = acwr(series, acute_days=7, chronic_days=28)
    assert abs(output[start + timedelta(days=27)] - 1.0) < 1e-6


def test_efficiency_index() -> None:
    eff, drift = efficiency_index(distance_m=10000, moving_time_s=3000, avg_hr=150, hr_stream=[145] * 60 + [155] * 60)
    assert eff > 0
    assert drift > 0


def test_data_quality_checks() -> None:
    gps_drift = detect_gps_drift([1.0, 2.0, 8.0], [[0, 0], [0, 0], [0, 0]])
    hr_flags = detect_hr_anomalies([39, 39, 39])
    assert gps_drift is True
    assert hr_flags["out_of_range"] is True
    assert hr_flags["flatline"] is True
