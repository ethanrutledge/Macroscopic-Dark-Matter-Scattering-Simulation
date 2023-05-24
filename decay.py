# initial framework for decay sampling
# adapted from probability notes from berger
# assuming 1 particle decays to 2

import numpy as np

# ----------------DECAY TIME----------------------

# assuming single decay mode
# therefore only one decay rate
decay_rate = 5      # placeholder value

# to store sampled decay times
t_samples = []

while len(t_samples) < 1000:
    # y is the probability of decay at time t
    y = np.random.random()

    # solver for time t wrt sampled probability
    t = -(1/decay_rate) * np.log(1 - y)

    # in the probability example it had a part checking if
    # the points were in the disk function and if less than Pmax
    # not sure how to apply that to this framework or if it is necessary

    # store sampled point
    t_samples.append(t)

# -------------------DECAY ANGLE------------------------
# -----------------------PART 1-------------------------
# boost to rest (CM) frame of decaying particle

# initial momentum and energy
# placeholder values -> will come from outgoing particles of scattering code
p_i = 10        # not sure if this is magnitude or vector momentum
E_i = 10

boost_parameter = -p_i/E_i

# again i think this needs to be done using uproot

# ----------------------PART 2---------------------------
# assuming particle has no spin
# therefore outgoing particles have random angle b/c
# spherically symmetric therefore no preferred angle

# c and polar angle sampled uniformly
# c from -1 to 1
c = 2.0 * np.random.random() - 1.0
polar = 2.0 * np.pi * np.random.random()

# --------------------PART 3---------------------------
# boost back to lab frame using inverse boost parameter

inverse_boost_parameter = -boost_parameter

# need to use uproot for this
