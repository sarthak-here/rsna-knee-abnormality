from pathlib import Path

import numpy as np

import optimize_nested_blend as optimizer
from optimize_nested_blend import EXPECTED_ARMS, simplex_grid


def test_simplex_grid() -> None:
    grid = simplex_grid()
    assert grid.shape == (286, 4)
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert (grid >= 0).all()


def test_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(optimizer, "BOOTSTRAP_DRAWS", 25)
    rng = np.random.default_rng(17)
    truth = np.zeros((58, 12), dtype=np.int8)
    prediction = np.empty((58, 4, 12), dtype=np.float64)
    for target in range(12):
        truth[:, target] = np.asarray(([0, 1] * 29), dtype=np.int8)
        rng.shuffle(truth[:, target])
        signal = truth[:, target] + rng.normal(0, 0.45, 58)
        prediction[:, 0, target] = 1 / (1 + np.exp(-signal))
        prediction[:, 1, target] = rng.random(58)
        prediction[:, 2, target] = rng.random(58)
        prediction[:, 3, target] = rng.random(58)
    package = tmp_path / "cache.npz"
    np.savez_compressed(
        package,
        ids=np.asarray([f"study-{index:03d}" for index in range(58)]),
        targets=np.asarray([f"target-{index:02d}" for index in range(12)]),
        truth=truth,
        arm_names=np.asarray(EXPECTED_ARMS),
        predictions=prediction,
    )
    output = tmp_path / "report.json"
    report = optimizer.run(package, output)
    assert output.exists()
    assert report["nested_coverage_complete"] is True
    assert report["study_count"] == 58
    assert report["submission_created"] is False
