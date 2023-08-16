# Implementation of the DM-Scattering-Model-Classes script
import DMNC_Classes
import matplotlib.pyplot as plt

# create instance of detector with dimensions of dune
dune = DMNC_Classes.Detector(12, 14, 58.2)

# create instance of model with argon(MM, mass density, mass) and dune detector
model = DMNC_Classes.Model(39.948, 1.3982, 6.63352 * (10 ** -26), 18, dune)

# create instance of simulation for current model specs and 100 runs
simulation = DMNC_Classes.Simulation(model, 100)
simulation.run()
simulation.write()
momenta = simulation.read_momenta('photons_emitted.HepMC3')

# create plot from momenta
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))
energies = []
p_x = []
p_y = []
p_z = []
for j in range(len(momenta)):
    energies.append(momenta[j].E)
    p_x.append(momenta[j].x)
    p_y.append(momenta[j].y)
    p_z.append(momenta[j].z)

axs[0, 0].hist(energies)
axs[0, 0].set_title('energies')
axs[0, 1].hist(p_x)
axs[0, 1].set_title('x direction momentum')
axs[0, 2].hist(p_y)
axs[0, 2].set_title('y direction momentum')
axs[0, 3].hist(p_z)
axs[0, 3].set_title('z direction momentum')

plt.savefig('momenta-outgoing-photons-all')

