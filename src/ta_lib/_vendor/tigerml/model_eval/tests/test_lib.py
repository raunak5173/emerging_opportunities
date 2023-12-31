import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from tigerml.core.preprocessing import *
from unittest import TestCase

# FIXME: use hypothesis library to generate test data.
TEST_DATA = pd.DataFrame(
    {
        "integral": range(10),
        "text": ["\u018e", ""] + ["abc"] * 8,
        "timestamp1": (
            pd.date_range("2016-01-01", periods=10, freq="1D").strftime("%y-%m-%d")
        ),
        "timestamp2": pd.date_range("2016-01-01", periods=10, freq="1D"),
        "object": [object()] + [object()] * 9,
        "numeric": np.arange(10).astype(float),
        # 'list': [[1,2]] * 10,
    }
)


class TestLib(TestCase):
    """Test lib class initializer."""

    def test_handle_outliers(self):
        """Test lib class initializer."""
        return 0
