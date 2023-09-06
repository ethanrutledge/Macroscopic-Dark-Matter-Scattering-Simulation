# Implementation of the DM-Scattering-Model-Classes script
import DMNC_Classes
import matplotlib.pyplot as plt

# create instance of model with dimensions of dune, and argon as material (A, Z, molar mass (g/mol), mass density (Mg/m^3))
model = DMNC_Classes.Model([12, 14, 58.2], 18, 40, 39.948, 1.3982)

# create instance of simulation for current model specs and 100 runs
simulation = DMNC_Classes.Simulation(model, 10)
simulation.run()
simulation.write()

momenta = simulation.read_momenta('photons_emitted.HepMC3')

