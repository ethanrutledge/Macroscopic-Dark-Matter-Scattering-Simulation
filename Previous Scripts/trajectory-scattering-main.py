# completed DM-trajectory code with scattering added (in progress)

import numpy as np
import uproot_methods as urm
import matplotlib.pyplot as plt

# -------------------------DETECTOR GEOMETRY--------------------------------
# based on dimensions of DUNE detector -- sourced from TDR doc
# using scale of 12 x 14 x 58.2m centered around the origin
x_scale = 12
y_scale = 14
z_scale = 58.2

# calculate the points of each face based on scale
x_max = x_scale / 2
x_min = -x_scale / 2
y_max = y_scale / 2
y_min = -y_scale / 2
z_max = z_scale / 2
z_min = -z_scale / 2

# r_max
# must be large enough s.t. no matter disk orientation target is covered
r_max = np.sqrt((x_scale ** 2) + (y_scale ** 2) + (z_scale ** 2))/2

# ---------------------SAMPLE TRAJECTORIES-----------------------------------
# store found entry and exit points
entries = []
exits = []

# store c and phis to check distribution
cs = []
phis = []

# store velocity magnitudes to check distribution
speeds = []

# iterate for some number of samples of trajectories
for j in range(100):
    # ------------------------DM DISK ORIENTATION------------------------
    # sample c uniformly from -1 to 1
    c = 2.0 * np.random.random() - 1.0
    cs.append(c)
    # convert c to theta(azimuthal angle)
    theta = np.arccos(c)

    # sample phi(polar angle) uniformly from 0 to 2pi
    phi = 2.0 * np.pi * np.random.random()
    phis.append(phi)

    # sample alpha(angle about disk) uniformly from 0 to 2pi
    alpha = 2.0 * np.pi * np.random.random()

    # sample probability of radius uniformly from 0 to 1
    prob_r = np.random.random()
    # calculate radius from probability
    r = r_max * np.sqrt(prob_r)

    # ----------------------TRAJECTORY CALCULATION--------------------------
    # initial disk orientation, fixed to z axis
    disk = urm.TVector3(np.cos(alpha), np.sin(alpha), 0)
    # disk oriented by theta around y-axis then by phi around z-axis
    disk = disk.rotatey(theta)
    disk = disk.rotatez(phi)
    # resize disk to radius
    disk = r * disk

    # trajectory: <x, y, z> = lambda * [orientation] + [disk]
    orient = [np.sqrt(1 - c ** 2) * np.cos(phi), np.sqrt(1 - c ** 2) * np.sin(phi), c]

    # ---------------------ENTRY AND EXIT POINTS---------------------------
    # to store solved trajectories at each plane
    faces = []
    lambdas = np.zeros(6)
    # to store trajectories that are either entry or exit points
    valid_points = []
    valid_lambdas = []

    # for each of the 6 faces: solve lamda using fixed axis point then subsequently solve trajectory equation
    # face 0 -> xy plane at z_max
    lambdas[0] = (z_max - disk.z) / orient[2]
    faces.append(urm.TVector3(lambdas[0] * orient[0] + disk.x, lambdas[0] * orient[1] + disk.y, z_max))

    # face 1 -> xy plane at z_min
    lambdas[1] = (z_min - disk.z) / orient[2]
    faces.append(urm.TVector3(lambdas[1] * orient[0] + disk.x, lambdas[1] * orient[1] + disk.y, z_min))

    # face 2 -> zy plane at x_max
    lambdas[2] = (x_max - disk.x) / orient[0]
    faces.append(urm.TVector3(x_max, lambdas[2] * orient[1] + disk.y, lambdas[2] * orient[2] + disk.z))

    # face 3 -> zy plane at x_min
    lambdas[3] = (x_min - disk.x) / orient[0]
    faces.append(urm.TVector3(x_min, lambdas[3] * orient[1] + disk.y, lambdas[3] * orient[2] + disk.z))

    # face 4 -> xz plane at y_max
    lambdas[4] = (y_max - disk.y) / orient[1]
    faces.append(urm.TVector3(lambdas[4] * orient[0] + disk.x, y_max, lambdas[4] * orient[2] + disk.z))

    # face 5 -> xz plane at y_min
    lambdas[5] = (y_min - disk.y) / orient[1]
    faces.append(urm.TVector3(lambdas[5] * orient[0] + disk.x, y_min, lambdas[5] * orient[2] + disk.z))

    # check which (if any) points are valid points on the faces of the detector
    for i in range(6):
        if (faces[i].x <= x_max) & (faces[i].x >= x_min) & (faces[i].y <= y_max) & (faces[i].y >= y_min) & (
                faces[i].z <= z_max) & (faces[i].z >= z_min):
            valid_points.append(faces[i])
            valid_lambdas.append(lambdas[i])

    # add valid points to entries and exit lists if found
    if len(valid_points) == 2:
        # check which is entry and exit
        if valid_lambdas[0] < valid_lambdas[1]:
            entry = valid_points[0]
            exit = valid_points[1]
            entries.append(valid_points[0])
            exits.append(valid_points[1])
        else:
            entry = valid_points[1]
            exit = valid_points[0]
            entries.append(valid_points[1])
            exits.append(valid_points[0])
    elif len(valid_points) == 1 or len(valid_points) > 2:
        print('invalid number of trajectory entry/exit points')

    # ----------------------SAMPLE SPEED------------------------------------
    # v_bar most likely needs refined
    v_bar = np.c * (10 ** -3)

    # each component sampled from a normal distribution with mean 0 and standard deviation v_bar/sqrt(3)
    v_x = np.random.normal(0, v_bar/np.sqrt(3))
    v_y = np.random.normal(0, v_bar/np.sqrt(3))
    v_z = np.random.normal(0, v_bar/np.sqrt(3))

    # find magnitude of velocity
    speed = np.sqrt((v_x ** 2) + (v_y ** 2) + (v_z ** 2))
    speeds.append(speed)

    # create a normalized trajectory vector
    traj_vect = urm.TVector3(exit.x - entry.x, exit.y - entry.y, exit.z - entry.z)
    traj_vect_norm = traj_vect / np.sqrt((traj_vect.x ** 2) + (traj_vect.y ** 2) + (traj_vect.z ** 2))

    # construct velocity vector from trajectory and speed
    cur = len(exits)
    velocity = speed * traj_vect_norm

    # -----------------------------------------INTERACTION--------------------------------------
    # cross-section taken from figure 7 of DM nucleus capture paper
    cross_section = 10 ** -25

    # radiative capture radius
    # derived from eq 32 of DM nucleus capture paper
    r_phi = (((cross_section * speed)/(60 * (10 ** -3))) ** 2) * (10 ** 5)

    # calculation of number density
    # molar mass of target -- in case of DUNE this is liquid argon
    molar_mass_target = 39.948  # g/mol
    # mass density of target -- in case of DUNE this is liquid argon
    mass_density_target = 1.3982    # g/L (at boiling point)
    # number density of target
    number_density_target = (np.Avogadro * mass_density_target) / molar_mass_target

    # to track total sampled distance along trajectory
    tot_dist = 0
    # to track all the interaction points sampled
    all_inter_pts = []

    # continue looping until reach break statement i.e. until sampled point leaves the detector
    while 1:
        # sample distance along trajectory until interaction
        y = np.random.random()
        dist = -(1/(number_density_target * cross_section)) * np.log(1 - y)
        tot_dist = tot_dist + dist

        # find point along trajectory from sampled distance
        inter_pt = entry + tot_dist * traj_vect_norm
        all_inter_pts.append(inter_pt)

        # if the interaction point is within the bounds of the detector
        if (inter_pt.x <= x_max) & (inter_pt.x >= x_min) & (inter_pt.y <= y_max) & (inter_pt.y >= y_min) & (inter_pt.z <= z_max) & (inter_pt.z >= z_min):
            # --------------------------------------SCATTERING-------------------------------------
            # ---------------------------PART 1-----------------------------------------
            # incoming beam particle and stationary target particle
            # construct 4 vector momentum using known energy and momentum

            # target particle (stationary)
            mass_target = 6.6335209 * (10 ** -26)    # in the case of DUNE this is argon
            target = urm.TLorentzVector(mass_target, 0, 0, 0)

            # beam particle (dark matter)
            mass_DM = (10 ** 20.5) * (1.7826 * (10 ** -27))   # converted from Gev to kg
            gamma = 1 / np.sqrt(1 - speed ** 2)
            p_beam = gamma * velocity
            p_mag_beam = np.sqrt((p_beam.x ** 2) + (p_beam.y ** 2) + (p_beam.z ** 2))
            beam = urm.TLorentzVector(np.sqrt((p_mag_beam ** 2) + mass_DM ** 2), p_beam.x, p_beam.y, p_beam.z)

            # ---------------------------PART 2---------------------------------------
            # apply lorentz boost s.t. sum of incoming four-momentum equals zero
            boost_factor = urm.TVector3(beam.x + target.x, beam.y + target.y, beam.z + target.z) / -(beam.E + target.E)

            target_boosted = target.boost(boost_factor)
            beam_boosted = beam.boost(boost_factor)

            # ---------------------------PART 3--------------------------------------
            # solve for 4 momenta for outgoing particles

            # the binding energy coming from the argon/DM bound state
            # this is currently approximated to be 1/r_phi but may need adjusted later on
            E_binding_bound_state = 1 / r_phi

            # masses of outgoing particles
            mass_out_photon = 0
            mass_out_DM = mass_DM + mass_target - E_binding_bound_state

            # total energy of center of mass frame
            E_cm = beam_boosted.E + target_boosted.E

            # energy of outgoing particles
            E_out_photon = (E_cm ** 2 + mass_out_photon ** 2 - mass_out_DM ** 2) / (2 * E_cm)
            E_out_DM = (E_cm ** 2 + mass_out_DM ** 2 - mass_out_photon ** 2) / (2 * E_cm)

            # solve for magnitudes of two momenta
            p_photon_abs = p_DM_abs = np.sqrt(E_out_photon ** 2 - mass_out_photon ** 2)

            # sample c uniformly from -1 to 1
            # could potentially be impacted by differential cross-section but will reassess this later on
            c_out = 2.0 * np.random.random() - 1.0
            # sample phi(polar angle) uniformly from 0 to 2pi
            phi_out = 2.0 * np.pi * np.random.random()

            # construct four-vector momenta in boosted frame
            out_photon_boosted = urm.TLorentzVector(E_out_photon, p_photon_abs * np.sqrt(1 - c_out ** 2) * np.cos(phi_out), p_photon_abs * np.sqrt(1 - c_out ** 2) * np.sin(phi_out), p_photon_abs * c_out)
            out_DM_boosted = urm.TLorentzVector(E_out_DM, -p_photon_abs * np.sqrt(1 - c_out ** 2) * np.cos(phi_out), -p_photon_abs * np.sqrt(1 - c_out ** 2) * np.sin(phi_out), -p_photon_abs * c_out)

            # ----------------------------------PART 4---------------------------------------
            # boost back to lab frame using inverse boost parameter
            inverse_boost_factor = -boost_factor

            out_photon = out_photon_boosted.boost(inverse_boost_factor)
            out_DM = out_DM_boosted.boost(inverse_boost_factor)
        else:
            break   # if the sampled point is not in the detector break the loop


