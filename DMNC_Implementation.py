# Implementation of the DM-Scattering-Model-Classes script

import DMNC_Classes

# create instance of detector with dimensions of dune
dune = DMNC_Classes.Detector(12, 14, 58.2)

# create instance of model with argon(MM, mass density, mass) and dune detector
model = DMNC_Classes.Model(39.948, 1.3982, 6.63352 * (10 ** -26), dune)

# create instance of simulation for current model specs and 100 runs
simulation = DMNC_Classes.Simulation(model, 100)

# create list of list for all trajectories for all photons from scattering and decay
scattered_photons, decay_photons = simulation.run()
