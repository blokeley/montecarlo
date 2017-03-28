"""Example of using the montecarlo module.

This example performs a Monte Carlo simulation for kinetic energy of a
moving body.
"""

import matplotlib.pyplot as plt

import montecarlo as mc


def energy(mass, velocity):
    return 0.5 * mass * velocity ** 2


# Create the mass and velocity parameters
masses = mc.Parameter('Mass', 10, 1)
masses.hist()
# If you are not using IPython, use plt.show() here

velocities = mc.Parameter('Velocity', 5, 0.2)
velocities.hist()

# Perform the Monte Carlo simulation on the random variates for each Parameter
energies = energy(masses.rvs, velocities.rvs)

# Plot them
plt.hist(energies, 100, normed=True)
plt.xlabel('Kinetic energy')
plt.ylabel('Probability density')
plt.tight_layout()
plt.show()

#  Calculate the parts per million above 135
print(mc.above(energies, 135))
