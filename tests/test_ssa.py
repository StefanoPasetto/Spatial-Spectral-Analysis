import unittest
import numpy as np

# Import the functions you want to test from your "ssa/ssa.py" module
from ssa.ssa import (
    two_point_correlation_function,
    compute_psd,
)


class TestTwoPointCorrelationFunction(unittest.TestCase):
    def test_uniform_distribution(self):
        """
        Generate a uniform random set of points. The 2pCF for a
        Poisson (uniform) distribution should be ~1 (no correlations).
        """
        np.random.seed(0)
        num_points = 500
        box_size = (100.0, 100.0)

        # Generate random points within [0, box_size[0]] x [0, box_size[1]]
        points = np.column_stack([
            np.random.uniform(0, box_size[0], num_points),
            np.random.uniform(0, box_size[1], num_points)
        ])

        r_min = 0.0
        r_max = min(box_size) / 2.0
        bins = 20

        # Call your function under test
        r_vals, xi_vals = two_point_correlation_function(points, r_min, r_max, bins, box_size)

        # For a large uniform sample, xi(r) should hover around 1
        mean_xi = np.mean(xi_vals)
        self.assertAlmostEqual(
            mean_xi, 1.0, delta=0.2,
            msg="Mean 2pCF for uniform distribution should be around 1"
        )

    def test_small_manual_points(self):
        """
        Test with a tiny, manually specified set of points
        to ensure no index errors and correct bin counts.
        """
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], dtype=float)

        box_size = (1.0, 1.0)
        r_min = 0.0
        r_max = 1.5
        bins = 5

        r_vals, xi_vals = two_point_correlation_function(points, r_min, r_max, bins, box_size)

        # Check bin count
        self.assertEqual(
            len(r_vals), bins,
            "Number of bin centers should match the requested bins."
        )
        # Check that 2pCF does not produce NaNs
        self.assertFalse(
            np.isnan(xi_vals).any(),
            "2pCF result should not contain NaNs."
        )


class TestPowerSpectralDensity(unittest.TestCase):
    def test_basic_psd(self):
        """
        Check that compute_psd() returns valid arrays (k_values, PSD_vals)
        and doesn't produce NaNs.
        """
        # Fake correlation data; enough variation to test the pipeline
        xi_vals = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])

        # Suppose we want to test PSD in the range 0.001 to 0.1
        k_values, psd_vals = compute_psd(xi_vals, klmin=0.001, klmax=0.1)

        self.assertEqual(
            len(k_values), len(psd_vals),
            "k_values and PSD array must have the same length."
        )
        self.assertFalse(
            np.isnan(psd_vals).any(),
            "PSD result should not contain NaNs."
        )
        self.assertTrue(
            np.all(psd_vals >= 0),
            "PSD should be non-negative or at least not negative. (Check or adapt if negative is possible in your theory.)"
        )


if __name__ == '__main__':
    unittest.main()
