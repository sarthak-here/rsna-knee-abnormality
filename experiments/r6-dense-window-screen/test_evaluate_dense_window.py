import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dense", HERE / "evaluate_dense_window.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_replacing_r6_with_identical_predictions_preserves_blend():
    rng = np.random.default_rng(17)
    predictions = rng.random((58, 4, 12))
    anchor = MOD.blend(predictions)
    candidate = predictions.copy()
    candidate[:, 0] = predictions[:, 0]
    assert np.array_equal(anchor, MOD.blend(candidate))


def test_blend_shape_and_finiteness():
    predictions = np.linspace(0, 1, 58 * 4 * 12).reshape(58, 4, 12)
    out = MOD.blend(predictions)
    assert out.shape == (58, 12)
    assert np.isfinite(out).all()
