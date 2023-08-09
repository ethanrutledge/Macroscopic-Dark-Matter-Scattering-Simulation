# Generic Classes Nucleus capture by Macroscopic DM, stores outgoing photon 4-momenta
import numpy as np
import uproot_methods as urm
import pyhepmc as hm


class Detector:
    # currently only works for rectangular detectors
    def __init__(self, x_scale, y_scale, z_scale):
        # calculate the points of each face based on scale
        self.x_max = x_scale / 2
        self.x_min = -x_scale / 2
        self.y_max = y_scale / 2
        self.y_min = -y_scale / 2
        self.z_max = z_scale / 2
        self.z_min = -z_scale / 2

        # r_max
        # must be large enough s.t. no matter disk orientation target is covered
        self.r_max = np.sqrt((x_scale ** 2) + (y_scale ** 2) + (z_scale ** 2)) / 2

    def within_bounds(self, pt):
        if (pt.x <= self.x_max) & (pt.x >= self.x_min) & (pt.y <= self.y_max) & (pt.y >= self.y_min) & (pt.z <= self.z_max) & (pt.z >= self.z_min):
            return 1


class Model:
    def __init__(self, molar_mass_target, mass_density_target, mass_target, detector):
        self.molar_mass_target = molar_mass_target
        self.mass_density_target = mass_density_target
        self.mass_target = mass_target
        self.detector = detector

    def cross_section(self):
        # cross-section currently taken from figure 7 of DM nucleus capture paper
        return 10 ** -25

    def decay_rate(self):
        # simply a random placeholder value until the actual function is determined
        return 5

    def trajectory_velocity(self):
        # ---------------------SAMPLE TRAJECTORIES-----------------------------------
        # store found entry and exit points
        entries = []
        exits = []

        # ------------------------DM DISK ORIENTATION------------------------
        # sample c uniformly from -1 to 1
        c = 2.0 * np.random.random() - 1.0
        # convert c to theta(azimuthal angle)
        theta = np.arccos(c)

        # sample phi(polar angle) uniformly from 0 to 2pi
        phi = 2.0 * np.pi * np.random.random()

        # sample alpha(angle about disk) uniformly from 0 to 2pi
        alpha = 2.0 * np.pi * np.random.random()

        # sample probability of radius uniformly from 0 to 1
        prob_r = np.random.random()
        # calculate radius from probability
        r = self.detector.r_max * np.sqrt(prob_r)

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
        lambdas[0] = (self.detector.z_max - disk.z) / orient[2]
        faces.append(urm.TVector3(lambdas[0] * orient[0] + disk.x, lambdas[0] * orient[1] + disk.y, self.detector.z_max))

        # face 1 -> xy plane at z_min
        lambdas[1] = (self.detector.z_min - disk.z) / orient[2]
        faces.append(urm.TVector3(lambdas[1] * orient[0] + disk.x, lambdas[1] * orient[1] + disk.y, self.detector.z_min))

        # face 2 -> zy plane at x_max
        lambdas[2] = (self.detector.x_max - disk.x) / orient[0]
        faces.append(urm.TVector3(self.detector.x_max, lambdas[2] * orient[1] + disk.y, lambdas[2] * orient[2] + disk.z))

        # face 3 -> zy plane at x_min
        lambdas[3] = (self.detector.x_min - disk.x) / orient[0]
        faces.append(urm.TVector3(self.detector.x_min, lambdas[3] * orient[1] + disk.y, lambdas[3] * orient[2] + disk.z))

        # face 4 -> xz plane at y_max
        lambdas[4] = (self.detector.y_max - disk.y) / orient[1]
        faces.append(urm.TVector3(lambdas[4] * orient[0] + disk.x, self.detector, lambdas[4] * orient[2] + disk.z))

        # face 5 -> xz plane at y_min
        lambdas[5] = (self.detector.y_min - disk.y) / orient[1]
        faces.append(urm.TVector3(lambdas[5] * orient[0] + disk.x, self.detector, lambdas[5] * orient[2] + disk.z))

        # check which (if any) points are valid points on the faces of the detector
        for i in range(6):
            if self.detector.within_bounds(faces[i]):
                valid_points.append(faces[i])
                valid_lambdas.append(lambdas[i])

        # add valid points to entries and exit lists if found
        if len(valid_points) == 2:
            # check which is entry and exit
            if valid_lambdas[0] < valid_lambdas[1]:
                # write to entry and exit variables and respective arrays
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
            # if there is only 1 or more than 2 entry/exit points print an error message and exit function
            print('invalid number of trajectory entry/exit points')
            return

        # ----------------------SAMPLE SPEED------------------------------------
        # v_bar most likely needs refined
        v_bar = np.c * (10 ** -3)

        # each component sampled from a normal distribution with mean 0 and standard deviation v_bar/sqrt(3)
        v_x = np.random.normal(0, v_bar / np.sqrt(3))
        v_y = np.random.normal(0, v_bar / np.sqrt(3))
        v_z = np.random.normal(0, v_bar / np.sqrt(3))

        # find magnitude of velocity
        speed = np.sqrt((v_x ** 2) + (v_y ** 2) + (v_z ** 2))

        # create a normalized trajectory vector
        traj_vect = urm.TVector3(exit.x - entry.x, exit.y - entry.y, exit.z - entry.z)
        traj_vect_norm = traj_vect / np.sqrt((traj_vect.x ** 2) + (traj_vect.y ** 2) + (traj_vect.z ** 2))

        # construct velocity vector from trajectory and speed
        cur = len(exits)
        velocity = speed * traj_vect_norm

        return entry, exit, speed, velocity, traj_vect_norm

    def scattering_interaction(self):
        # pull the needed properties of the trajectory and velocity from method trajectory_velocity
        entry, exit, speed, velocity, traj_vect_norm = self.trajectory_velocity()
        # pull the cross-section and decay rate from respective methods
        cross_section = self.cross_section()
        decay_rate = self.decay_rate()

        # -----------------------------------------INTERACTION--------------------------------------
        # radiative capture radius
        # derived from eq 32 of DM nucleus capture paper
        r_phi = (((cross_section * speed) / (60 * (10 ** -3))) ** 2) * (10 ** 5)

        # calculation of number density
        number_density_target = (np.Avogadro * self.mass_density_target) / self.molar_mass_target

        # to track total sampled distance along trajectory
        tot_dist = 0
        # to track all the interaction points sampled
        all_inter_pts = []

        # to store all photons emitted from scattering and decay respectively
        scattered_photons = np.array([])
        decay_photons = np.array([])

        # continue looping until reach break statement i.e. until sampled point leaves the detector
        while 1:
            # sample distance along trajectory until interaction
            y = np.random.random()
            dist = -(1 / (number_density_target * cross_section)) * np.log(1 - y)
            tot_dist = tot_dist + dist

            # find point along trajectory from sampled distance
            inter_pt = entry + tot_dist * traj_vect_norm
            all_inter_pts.append(inter_pt)

            # if the interaction point is within the bounds of the detector
            if self.detector.within_bounds(inter_pt):
                # --------------------------------------SCATTERING-------------------------------------
                # ---------------------------PART 1-----------------------------------------
                # incoming beam particle and stationary target particle
                # construct 4 vector momentum using known energy and momentum

                # target particle (stationary)
                mass_target = self.mass_target
                target = urm.TLorentzVector(mass_target, 0, 0, 0)

                # beam particle (dark matter)
                mass_DM = (10 ** 20.5) * (1.7826 * (10 ** -27))  # converted from Gev to kg
                gamma = 1 / np.sqrt(1 - speed ** 2)
                p_DM_in = gamma * velocity
                p_mag_DM_in = np.sqrt((p_DM_in.x ** 2) + (p_DM_in.y ** 2) + (p_DM_in.z ** 2))
                DM_in = urm.TLorentzVector(np.sqrt((p_mag_DM_in ** 2) + mass_DM ** 2), p_DM_in.x, p_DM_in.y, p_DM_in.z)

                # ---------------------------PART 2---------------------------------------
                # apply lorentz boost s.t. sum of incoming four-momentum equals zero
                boost_factor = urm.TVector3(DM_in.x + target.x, DM_in.y + target.y, DM_in.z + target.z) / -(
                            DM_in.E + target.E)

                target_boosted = target.boost(boost_factor)
                beam_boosted = DM_in.boost(boost_factor)

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
                p_photon_abs = np.sqrt(E_out_photon ** 2 - mass_out_photon ** 2)

                # sample c uniformly from -1 to 1
                # could potentially be impacted by differential cross-section but will reassess this later on
                c_out = 2.0 * np.random.random() - 1.0
                # sample phi(polar angle) uniformly from 0 to 2pi
                phi_out = 2.0 * np.pi * np.random.random()

                # construct four-vector momenta in boosted frame
                out_photon_boosted = urm.TLorentzVector(E_out_photon,
                                                        p_photon_abs * np.sqrt(1 - c_out ** 2) * np.cos(phi_out),
                                                        p_photon_abs * np.sqrt(1 - c_out ** 2) * np.sin(phi_out),
                                                        p_photon_abs * c_out)
                out_DM_boosted = urm.TLorentzVector(E_out_DM, -p_photon_abs * np.sqrt(1 - c_out ** 2) * np.cos(phi_out),
                                                    -p_photon_abs * np.sqrt(1 - c_out ** 2) * np.sin(phi_out),
                                                    -p_photon_abs * c_out)

                # ----------------------------------PART 4---------------------------------------
                # boost back to lab frame using inverse boost parameter
                inverse_boost_factor = -boost_factor
                out_photon = out_photon_boosted.boost(inverse_boost_factor)
                out_DM = out_DM_boosted.boost(inverse_boost_factor)

                # create vertex with scattered photon and add to new scattering event
                event_scatter = hm.GenEvent()
                vertex_scatter = hm.GenVertex()
                scattered_photon = hm.GenParticle()
                scattered_photon.set_pdg_id(22)
                scattered_photon.set_momentum(out_photon)
                vertex_scatter.set_position(inter_pt)   # this needs a time element but not sure where comes from
                vertex_scatter.add_particle_out(hm.FourVector(out_photon.E, out_photon.x, out_photon.y, out_photon.z))
                event_scatter.add_vertex(vertex_scatter)

                # append the scattered photon event to array of all scattered photons along current trajectory
                np.append(scattered_photons, event_scatter)

                # ---------------------------------DECAY-------------------------------------------
                # --------------INCOMPLETE SECTION STILL IN PROGRESS-------------------------------
                # temporarily used values NEED REVISED
                current_state_E = out_DM.E
                ground_state_E = target.E

                # create decay event
                event_decay = hm.GenEvent()

                while 1:
                    if current_state_E > ground_state_E:
                        # decay time
                        tr = np.random.random()
                        decay_time = (-1 / decay_rate) * np.log(1 - tr)

                        # create vertex at current time and interation point
                        vertex_decay = hm.GenVertex()
                        # -------------------------------------THE TIME HERE IS NOT CORRECT NEEDS TO BE TOTAL TIME
                        vertex_decay.set_position(hm.FourVector(decay_time, inter_pt.x, inter_pt.y, inter_pt.z))

                        # assuming there is no spin there is no preferred direction so c and phi can be sampled uniformly
                        # sample c uniformly from -1 to 1
                        c_decay = 2.0 * np.random.random() - 1.0
                        # sample phi(polar angle) uniformly from 0 to 2pi
                        phi_decay = 2.0 * np.pi * np.random.random()

                        # calculate momentum magnitude and energy of decay particle
                        # it is assumed for now that the emitted photon has energy of 1 / r-phi
                        E_decay_photon = 1 / r_phi
                        p_decay_photon = np.sqrt(E_decay_photon ** 2 - mass_out_photon ** 2)

                        # generate decay particle
                        decay_photon = hm.GenParticle()
                        decay_photon.set_pdg_id(22)
                        decay_photon.set_momentum(hm.FourVector(E_decay_photon, p_decay_photon * np.sqrt(1 - c_decay ** 2)
                                                                * np.cos(phi_decay), p_decay_photon * np.sqrt(1 - c_decay ** 2)
                                                                * np.sin(phi_decay), p_decay_photon * c_decay))

                        # add current decay to the decay event
                        vertex_decay.add_particle_out(decay_photon)
                        event_decay.add_vertex(vertex_decay)
                    else:
                        break
                # add decay event to array of all decays along trajectory
                np.append(decay_photons, event_decay)
            else:
                break  # if the sampled point is not in the detector break the loop

        return scattered_photons, decay_photons


class Simulation:
    def __init__(self, model, iterations):
        self.decay_photons = None
        self.scattered_photons = None
        self.model = model
        self.iterations = iterations

    def run(self):
        # arrays to store all photons from all trajectories
        scattered_photons_all = np.array([])
        decay_photons_all = np.array([])

        # iterates the scattering_interation method of the model for given number of trajectory attempts
        for n in range(self.iterations):
            # stores photons from specific trajectory
            scattered_photons, decay_photons = self.model.scattering_interaction()
            np.append(scattered_photons_all, scattered_photons)
            np.append(decay_photons_all, decay_photons)

        # stores filled array of photons emitted as property of the class
        self.scattered_photons = scattered_photons_all
        self.decay_photons = decay_photons_all

    def write(self, emission_type):
        # creates file to store events to
        writer = hm.WriterAscii('photons_emitted.HepMC3')

        # checks which emission type file is and pulls photon array respectively
        if emission_type == 'scattered':
            photons = self.scattered_photons
        elif emission_type == 'decay':
            photons = self.decay_photons
        else:
            print('incorrect emission type')
            return

        # iterates through all trajectories and all events writing each to the writer
        for i in range(len(photons)):
            this_traj = photons[i]
            for j in range(len(this_traj)):
                writer.write_event(this_traj[j])
