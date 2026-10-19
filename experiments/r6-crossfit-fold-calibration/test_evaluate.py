import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("r6_calibration", MODULE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_fixed_combiner_uses_all_eight_logits():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(500, 8))
    soft = MOD.sigmoid(1.2 * x[:, 0] - .8 * x[:, 5] + .15 * rng.normal(size=500))
    prediction, receipt = MOD.fit_predict(x[:400], soft[:400], x[400:])
    assert prediction.shape == (100,)
    assert np.isfinite(prediction).all()
    assert np.all((prediction >= 0) & (prediction <= 1))
    assert len(receipt["coef"]) == 8
    assert receipt["alpha"] == 100.0


def test_logit_sigmoid_roundtrip():
    values = np.asarray([.01, .2, .5, .8, .99])
    assert np.allclose(MOD.sigmoid(MOD.logit(values)), values)
