"""Utility classes and functions for Monte Carlo simulation.

Copyright 2016 - 2017 Tom Oakley
MIT licence: https://opensource.org/licenses/MIT

Ask Tom Oakley for advice before modifying this file.
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


__version__ = '1.0'


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

    def __init__(self, name, target, tolerance):
        """name(str): a string identifier
        target(float): the target value (often the mean)
        tolerance(float): the symmetrical tolerance.  For asymmetrical
            tolerances, set the random variates (rvs property) directly
        """
        self.name = name
        self.lsl = target - tolerance
        self.target = target
        self.usl = target + tolerance

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
            std = (self.usl - self.lsl) / (6 * CP)
            # Create random variates (rvs)
            self.rvs = norm.rvs(self.target, std, TRIALS)
            return self._rvs

    @rvs.setter
    def rvs(self, variates):
        """Set the random variates (samples from the distribution)."""
        # Check that the type and size of the variates array is correct
        if not (isinstance(variates, np.ndarray) and variates.size == TRIALS):
                fmt = 'variates must be an ndarray of size {}'
                raise ValueError(fmt.format(TRIALS))

        self._rvs = variates

    def hist(self):
        """Plot a histogram and vertical lines for lower and upper
        specification limits and target if they are given.
        """
        plt.hist(self.rvs, 100, normed=True)
        plt.axvline(self.lsl, color='r')
        plt.axvline(self.usl, color='r')
        plt.axvline(self.target, color='g')
        plt.xlabel(self.name)
        plt.ylabel('Probability density')
        plt.show()


def above(arr, maximum):
    """Return the parts per million in array arr above maximum."""
    return 1e6 * arr[arr > maximum].size / arr.size


def below(arr, minimum):
    """Return the parts per million in array arr below minimum."""
    return 1e6 * arr[arr < minimum].size / arr.size


def describe(results, units='', lsl=None, usl=None):
    """Print useful statistics about the results."""
    print('Mean = {:.3f} {}'.format(results.mean(), units))
    # Use 1 degree of freedom because this is a sample, not a population
    print('Standard deviation = {:.3f} {}'.format(results.std(ddof=1), units))

    if lsl is not None:
        print('{:.0f} ppm below {} {}'.format(below(results, lsl), lsl, units))

    if usl is not None:
        print('{:.0f} ppm above {} {}'.format(above(results, usl), usl, units))


# Unit tests
class TestParameter(unittest.TestCase):

    def test_rvs(self):
        target = 20
        tol = 2
        places = 2  # Resolution in decimal places
        p = Parameter('My parameter', target, tol)

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
