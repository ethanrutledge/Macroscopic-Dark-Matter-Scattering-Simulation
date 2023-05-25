# initial framework for probability sampling of interation between DM and target
# adapted from probability notes from berger
# did particle interact at point x?

import numpy as np

# number density of target particles
n = 10         # placeholder value

cross_section = 10**-24     # placeholder value

# will store sampled interaction points
x_samples = []

# iterate for 1000 samples
while len(x_samples) < 1000:
    # y is probability of interaction at point x
    y = np.random.random()

    # solve for point x wrt sampled probability
    x = -(1/(n * cross_section)) * np.log(1 - y)

    # in the probability example it had a part checking if
    # the points were in the disk function and if less than Pmax
    # not sure how to apply that to this framework or if it is necessary

    # store sampled point
    x_samples.append(x)
