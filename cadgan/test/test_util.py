"""
Module for testing cadgan.util .
"""

__author__ = "wittawat"

import unittest

import cadgan.util as util
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as testing
# Import all the submodules for testing purpose
import scipy.stats as stats


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def test_linear_range_transform(self):
        A = np.array([0.1, 0.8])
        testing.assert_almost_equal(np.array([1, 8]), util.linear_range_transform(A, (0, 1), (0, 10)))

        testing.assert_almost_equal(
            np.array([-2, 3]), util.linear_range_transform(np.array([-0.2, 0.3]), (-1, 1), (-10, 10))
        )

    def test_dict_to_string(self):
        d = {"a": 0, "c": 2, "d": 3}
        testing.assert_equal("a0_c2_d3", util.dict_to_string(d, order=["a", "c", "d"], entry_sep="_", kv_sep=""))

        testing.assert_equal(
            "a0_c2_d3", util.dict_to_string(d, order=["a", "sad", "c", "d", "c"], entry_sep="_", kv_sep="")
        )

        testing.assert_equal(
            "a*0_c*2_d*3", util.dict_to_string(d, order=["a", "sad", "c", "d", "c"], entry_sep="_", kv_sep="*")
        )

        testing.assert_equal(
            "a*0-c*2-d*3", util.dict_to_string(d, order=["a", "sad", "c", "d", "c"], entry_sep="-", kv_sep="*")
        )
        testing.assert_equal(
            "a*0-d*3",
            util.dict_to_string(d, order=["a", "sad", "c", "d", "c"], exclude=["c"], entry_sep="-", kv_sep="*"),
        )

    def test_translate_keys(self):
        import copy

        d = {"a": 0, "c": 2}
        d2 = copy.deepcopy(d)
        util.translate_keys(d2, {"a": "aa", "c": "cc"})
        testing.assert_equal({"aa": 0, "cc": 2}, d2)

        d2 = copy.deepcopy(d)
        util.translate_keys(d2, {"a": "a", "d": "cc"})
        testing.assert_equal({"a": 0, "c": 2}, d2)

        d2 = copy.deepcopy(d)
        util.translate_keys(d2, {"8": "a", "ac": "cc"})
        testing.assert_equal({"a": 0, "c": 2}, d2)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
