import numpy as np

from morphology_features import pool_curve, slice_descriptors, study_features


def test_shape_and_determinism():
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, (6, 12, 32, 32), dtype=np.uint8)
    masks = np.array([1, 1, 0, 1, 0, 1], np.float32)
    first = study_features(pixels, masks)
    second = study_features(pixels, masks)
    assert first.shape == (1356,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_constant_image_has_zero_texture_and_slope():
    image = np.full((12, 24, 24), 128, np.uint8)
    descriptor = slice_descriptors(image)
    pooled = pool_curve(descriptor)
    assert descriptor.shape == (12, 30)
    assert np.allclose(descriptor[:, 7:12], 0.0)
    assert np.allclose(pooled[-30:], 0.0, atol=1e-7)


def test_missing_slot_is_explicit_and_zero_filled():
    pixels = np.full((6, 12, 24, 24), 128, np.uint8)
    masks = np.array([1, 0, 1, 1, 1, 1], np.float32)
    features = study_features(pixels, masks)
    slot_width = 150
    assert np.all(features[slot_width:2 * slot_width] == 0)
    assert np.array_equal(features[-6:], masks)
