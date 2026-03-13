"""The CI gate is only worth having if it actually fails when the detectors regress."""
import sys

import pytest

from evaluation import run_eval
from datascout.tools import quality


def run(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])
    code = run_eval.main()
    return code, capsys.readouterr().out


def test_gate_passes_on_current_detectors(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys)
    assert code == 0, out


@pytest.mark.parametrize("attr,bad_value,expected_failure", [
    ("MISSING_FLAG_PCT", 50.0, "recall"),            # detector goes blind -> recall drops
    ("LEAK_ETA", 0.5, "precision"),                   # detector too eager -> false positives
    ("COLLINEAR_R", 0.3, "precision"),
])
def test_gate_fails_when_detector_regresses(monkeypatch, capsys, attr, bad_value, expected_failure):
    monkeypatch.setattr(quality, attr, bad_value)
    code, out = run(monkeypatch, capsys)
    assert code == 1, f"gate did not fail after breaking {attr}\n{out}"
    assert f"{expected_failure} >=" in out and "FAIL" in out
