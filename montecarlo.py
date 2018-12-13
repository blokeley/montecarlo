"""Utility classes and functions for Monte Carlo simulation.

Copyright 2016 - 2017 Tom Oakley
MIT licence: https://opensource.org/licenses/MIT

Get the latest version from:
https://github.com/blokeley/montecarlo

Ask Tom Oakley for advice before modifying this file.

Versions:
1.0 - First use
1.1 - Add legend to histrogram.
      Allow asymmetric tolerances in Parameter.rvs
2.0 - Make Parameter name optional
2.0.1 - Do not use deprecated normed argument to plt.hist()
2.0.2 - Fix typo so self.rvs reads self._rvs
2.0.3 - Add string representation of parameter
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


__version__ = '2.0.2'


TRIALS = 1000000
"""Number of samples for Monte Carlo simulation"""

# It is general practice to require Cp and Cpk of at least 1.33.  Sources:
# http://www.qualitytrainingportal.com/resources/spc/spc_process_capability_measuring.htm
# https://en.wikipedia.org/wiki/Process_capability_index
# http://www.npd-solutions.com/proccap.html
# http://www.stat.purdue.edu/~jdobbin/stat301T/admin/ProcessCapabilitySpr07RevC.pdf
# http://sixsigmastudyguide.com/process-capability-cp-cpk/
#
# This model considers only centred distributions, so we cannot account for
# Cpk.  Therefore, the calculation uses cp only.
CP = 1.33
"""Default process capability"""


class Parameter:
    """Encapsulation of parameter properties such as specification limits and
    methods such as plotting.
    """

    def __init__(self, target, tolerance, name=''):
        """
        Args:
            target: the target value (often the mean)
            tolerance: the symmetrical tolerance.  For asymmetrical
                tolerances, set the random variates (rvs property) directly
            name: a string identifier
        """
        self.lsl = target - tolerance
        self.target = target
        self.usl = target + tolerance
        self.name = name

    @property
    def rvs(self):
        """Return random variates (samples from the distribution).

        Defaults to the normal distribution: patch or override for other
        distributions.
        """
        # If the random variates have already been created, return them
        try:
            return self._rvs

        # Else create the random variates.  This is lazy initialisation
        except AttributeError:
            # Calculate standard deviation
            # http://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm
            std = min(self.usl - self.target, self.target - self.lsl) / (3*CP)
            # Create random variates (rvs)
            self._rvs = norm.rvs(self.target, std, TRIALS)
            return self._rvs

    @rvs.setter
    def rvs(self, variates):
        """Set the random variates (samples from the distribution)."""
        # Check that the type and size of the variates array is correct
        if not (isinstance(variates, np.ndarray) and variates.size == TRIALS):
                fmt = 'variates must be an ndarray of size {}'
                raise ValueError(fmt.format(TRIALS))

        self._rvs = variates

    def hist(self, **kwargs):
        """Create a histogram and vertical lines for lower and upper
        specification limits and target.
        """
        fig, ax = plt.subplots()
        ax.hist(self.rvs, 100, density=True, label=self.name)
        ax.axvline(self.lsl, color='m', label='Lower spec limit')
        ax.axvline(self.usl, color='r', label='Upper spec limit')
        ax.axvline(self.target, color='g', label='Target')
        ax.set(xlabel=self.name, ylabel='Probability density')
        ax.legend()
        return ax

    def __str__(self):
        return 'Parameter {} {:.2f}'.format(self.name, self.target)


def above(arr, maximum):
    """Return the parts per million in array arr above maximum."""
    return 1000000 * arr[arr > maximum].size / arr.size


def below(arr, minimum):
    """Return the parts per million in array arr below minimum."""
    return 1000000 * arr[arr < minimum].size / arr.size


def describe(results, units='', lsl=None, usl=None):
    """Return useful statistics as a pandas.Series."""
    res = pd.Series()
    res['Median ({})'.format(units)] = np.median(results)
    res['Mean ({})'.format(units)] = results.mean()

    # Use 1 degree of freedom because this is a sample, not a population
    res['Standard deviation ({})'.format(units)] = results.std(ddof=1)

    if lsl is not None:
        res['ppm below {} {}'.format(lsl, units)] = below(results, lsl)

    if usl is not None:
        res['ppm above {} {}'.format(usl, units)] = above(results, usl)

    return res


# Unit tests
class TestParameter(unittest.TestCase):

    def test_rvs(self):
        target = 20
        tol = 2
        places = 2  # Resolution in decimal places
        p = Parameter(target, tol)

        #  Test mean is near target
        self.assertAlmostEqual(target, p.rvs.mean(), places)

        # Test standard deviation is approximately correct
        expected = tol / (3 * CP)
        self.assertAlmostEqual(expected, p.rvs.std(ddof=1), places)


class TestAbove(unittest.TestCase):

    def test_above(self):
        # Array of 1000 equispaced numbers
        arr = np.linspace(0, 1000, 1000)

        # Assert ppm above 900.  100 samples in 1000 is 100000
        self.assertEqual(100000, above(arr, 900))


if __name__ == '__main__':
    unittest.main()
