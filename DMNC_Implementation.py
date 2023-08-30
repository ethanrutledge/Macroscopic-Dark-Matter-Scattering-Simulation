# Implementation of the DM-Scattering-Model-Classes script
import DMNC_Classes
import matplotlib.pyplot as plt

# create instance of detector with dimensions of dune
dune = DMNC_Classes.Detector(12, 14, 58.2)

# create instance of model with argon(molar mass (g/mol), mass density (Mg/m^3), A, Z) and dune as detector
model = DMNC_Classes.Model(39.948, 1.3982, 40, 18, dune)

# create instance of simulation for current model specs and 100 runs
simulation = DMNC_Classes.Simulation(model, 100)
simulation.run()
simulation.write()

momenta = simulation.read_momenta('photons_emitted.HepMC3')

