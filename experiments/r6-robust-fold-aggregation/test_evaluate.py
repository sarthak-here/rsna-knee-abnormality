import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("robust", HERE / "evaluate.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_trimmed_fold_reducer_rejects_one_extreme_fold():
    r6 = np.full((58, 12), 0.5)
    logits = np.zeros((58, 4, 2, 12))
    logits[:, 3] = 20.0
    raptor = np.full((58, 2, 12), 0.5)
    _, candidate, reconstruction = MOD.predictions(r6, logits, raptor)
    assert reconstruction > 0.4
    assert np.isfinite(candidate).all()
    assert candidate.shape == (58, 12)


def test_rank_columns_shape_and_bounds():
    x = np.arange(120, dtype=float).reshape(10, 12)
    ranked = MOD.rank_columns(x)
    assert ranked.shape == x.shape
    assert ranked.min() > 0
    assert ranked.max() <= 1
