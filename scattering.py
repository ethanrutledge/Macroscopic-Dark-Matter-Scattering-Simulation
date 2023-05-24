# initial framework for scattering of DM and target particle
# adapted from probability notes from berger
# assuming particles definitely will interact
# assuming two incoming particles and two outgoing particles (could be same or different from incoming)
# assuming c = 1

import numpy as np

# assuming single scattering mode
# therefore only one cross-section
cross_section = 10  # placeholder value

# ---------------------------PART 1-----------------------------------------
# incoming beam particle and stationary target particle
# both have known energy and momentum -> construct 4 vector momentum

# target particle (stationary)
mass_target = 10  # placeholder value
E_target = mass_target  # assuming c = 1
p_target = [E_target, 0, 0, 0]

# beam particle (dark matter)
# does flux not need to be considered here?
mass_DM = 10  # placeholder value
beta_beam = 0.2  # placeholder value
lorentz_factor_beam = 1 / np.sqrt(1 - beta_beam ** 2)
v_beam = [1, 0.1, 0.1, 0.1]  # placeholder values (v0 is 1 because =c & assuming c = 1)
p_beam = lorentz_factor_beam * v_beam
E_beam = p_beam(0)

# ---------------------------PART 2---------------------------------------
# apply lorentz boost s.t. sum of incoming four-momentum equals zero

boost_factor = -(p_target + p_beam) / (E_target + E_beam)

# need to use uproot package for lorentz transform
# unsure how to do this

# ---------------------------PART 3--------------------------------------
# solve for 4 momenta for outgoing particles

mass_out_1 = 10  # placeholder value
mass_out_2 = 10  # placeholder value

# total energy of center of mass frame
E_cm = E_beam + E_target

# energy of outgoing particles
E_out_1 = (E_cm ** 2 + mass_out_1 ** 2 - mass_out_2 ** 2) / (2 * E_cm)
E_out_2 = (E_cm ** 2 + mass_out_2 ** 2 - mass_out_1 ** 2) / (2 * E_cm)

# in probability notes it is set up this way where magnitude of outgoing momentum is equal
# this does not totally make sense
p_1_abs = p_2_abs = np.sqrt(E_out_1 ** 2 - mass_out_1 ** 2)

# sampling of a polar and c angles
# c = cos(theta)

differential_cross_section = 10  # placeholder value

# probability of c and polar angles as defined by differential cross-section
def P(c, polar):
    return (1 / polar) * differential_cross_section


c_samples = []
polar_samples = []

Pmax = 1        # placeholder value

while len(c_samples) < 1000:

    # sample c uniformly from -1 to 1
    c = 2.0 * np.random.random() - 1.0
    # sample polar angle uniformly from 0 to 2pi
    polar = 2.0 * np.pi * np.random.random()

    # in the probability example it had a part checking if
    # the points were in the disk function, not sure how to
    # carry that over to this framework

    # check if P < Pmax
    if (P(c, polar) / Pmax) < np.random.random():
        continue

    # add to samples list
    c_samples.append(c)
    polar_samples.append(polar)

# construct four-vector momenta in boosted frame
# these equations have polar and c angles in them
# im not sure how to implement this as the sampled angles
# are stored as array not a scalar

p_1 = [E_out_1, p_1_abs * np.sqrt(1 - c_samples**2) * np.cos(polar_samples),
       p_1_abs * np.sqrt(1 - c_samples**2) * np.sin(polar_samples), p_1_abs * c_samples]

p_2 = [E_out_1, -p_1_abs * np.sqrt(1 - c_samples**2) * np.cos(polar_samples),
       -p_1_abs * np.sqrt(1 - c_samples**2) * np.sin(polar_samples), -p_1_abs * c_samples]

# ----------------------------------PART 4---------------------------------------
# boost back to lab frame
# using inverse boost parameter

inverse_boost_factor = -boost_factor

# again this lorentz transform needs to be done using uproot
# not sure how to do this/use uproot
