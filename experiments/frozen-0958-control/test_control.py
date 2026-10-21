import unittest
import numpy as np
from control import blend, RAPTOR_WEIGHTS


class ControlTests(unittest.TestCase):
    def test_ties_and_row_permutation(self):
        a = np.tile(np.array([.2, .2, .8])[:, None], (1, 12))
        b = a[::-1].copy()
        c = np.tile(np.array([.3, .6, .9])[:, None], (1, 12))
        result = blend(a, b, c)
        order = np.array([2, 0, 1])
        np.testing.assert_array_equal(result[order], blend(a[order], b[order], c[order]))
        expected_first = RAPTOR_WEIGHTS * (.65 * 1 + .35 / 3) + (1 - RAPTOR_WEIGHTS) * .5
        np.testing.assert_array_equal(result[0], expected_first.astype(np.float32))

    def test_reject_bad_readers(self):
        a = np.full((3, 12), .2)
        for b in (np.full((3, 12), np.nan), np.full((3, 12), 1.1), np.zeros((2, 12)), np.full((3, 12), .5)):
            with self.assertRaises(ValueError):
                blend(a, b, a)

    def test_constant_ties_use_average_not_ordinal(self):
        a = np.full((3, 12), .2)
        np.testing.assert_array_equal(blend(a, a, a), np.full((3, 12), 2 / 3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
