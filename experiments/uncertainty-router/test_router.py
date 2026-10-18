import numpy as np

from evaluate_uncertainty_router import build_predictions


def test_shapes_bounds_and_determinism():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(58, 4, 2, 12))
    raptor = rng.uniform(0.01, 0.99, size=(58, 2, 12))
    a1, c1, w1 = build_predictions(logits, raptor)
    a2, c2, w2 = build_predictions(logits, raptor)
    assert a1.shape == c1.shape == w1.shape == (58, 12)
    assert np.allclose(a1, a2) and np.allclose(c1, c2) and np.allclose(w1, w2)
    assert np.isfinite(a1).all() and np.isfinite(c1).all()
    assert w1.min() >= 0.45 and w1.max() <= 0.55


def test_rejects_bad_shapes():
    logits = np.zeros((58, 4, 12))
    raptor = np.zeros((58, 2, 12))
    try:
        build_predictions(logits, raptor)
    except ValueError as exc:
        assert "[N,4,2,12]" in str(exc)
    else:
        raise AssertionError("bad shape was accepted")
