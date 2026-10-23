import json

import pytest

from kernelbench.benchmark import comparison, parse_shape, quantile, run_cpu_validation, summarize


def test_quantiles_interpolate_and_preserve_input():
    values = [4., 1., 3., 2.]
    assert quantile(values, .5) == 2.5
    assert quantile(values, .1) == 1.3
    assert values == [4., 1., 3., 2.]
    assert summarize([2.]) == {"samples": 1, "p10_ms": 2., "median_ms": 2., "p90_ms": 2.}


@pytest.mark.parametrize("values,fraction", [([], .5), ([1.], -1), ([float("nan")], .5), ([-1.], .5)])
def test_bad_samples(values, fraction):
    with pytest.raises(ValueError):
        quantile(values, fraction)


def test_outcomes_report_losses_too():
    assert comparison(2, 1) == {"speedup": 2, "outcome": "win"}
    assert comparison(1, 2)["outcome"] == "loss"
    assert comparison(1, 1.01)["outcome"] == "within_5_percent"
    with pytest.raises(ValueError):
        comparison(1, 0)
    with pytest.raises(ValueError):
        comparison(float("nan"), 1)


def test_cpu_report_cannot_be_mistaken_for_gpu_results():
    report = run_cpu_validation()
    assert report["status"] == "cpu_reference_validation_only"
    assert report["gpu_status"] == "not_executed"
    assert len(report["results"]) == 4
    assert "median_ms" not in json.dumps(report, allow_nan=False)
    assert "speedup" not in json.dumps(report)


def test_parse_shape():
    assert parse_shape("128x781") == (128, 781)
    with pytest.raises(Exception):
        parse_shape("0x12")
