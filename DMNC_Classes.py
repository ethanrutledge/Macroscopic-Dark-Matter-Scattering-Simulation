import uproot_methods as urm
import pyhepmc as hm
import DMNC_Rates as rates
import numpy as np
# Bessel functions appear as the wavefunctions here
from scipy.special import spherical_jn,spherical_yn,jv,jvp,yvp
# Numerical integration
from scipy.integrate import quad
# Non-linear equation solver
from scipy.optimize import fsolve
# scipy can't do real order bessel function zeroes, so use mpmath
from mpmath import besseljzero
# binomial coefficients that appear in spherical harmonics products
from scipy.special import comb


class Model:
    def __init__(self, scale, atomic_number, mass_number, molar_mass, mass_density, R_phi = 10.0, mass_DM = 10 ** 25):
        self.x_max = scale(0) / 2                                                               # max x dimension of detector, m
        self.x_min = -scale(0) / 2                                                              # min x dimension of detector, m
        self.y_max = scale(1) / 2                                                               # max y dimension of detector, m
        self.y_min = -scale(1) / 2                                                              # min y dimension of detector, m
        self.z_max = scale(2) / 2                                                               # max z dimension of detector, m
        self.z_min = -scale(2) / 2                                                              # min z dimension of detector, m

        self.molar_mass_target = molar_mass                                                     # material molar mass
        self.mass_density_target = mass_density                                                 # material mass density
        self.mass_number_target = mass_number                                                   # material mass number, dimensionless
        self.atomic_number_target = atomic_number                                               # material atomic number, dimensionless
        self.number_density_target = ((6.0221 * (10 ** 23)) * mass_density) / molar_mass        # number density of target material

        self.electric_charge = np.sqrt(4.0 * np.pi / 137)                                       # electric charge, dimensionless
        self.V0 = mass_number * 0.246                                                           # potential depth, GeV
        self.R_phi = R_phi                                                                      # DM radiative capture radius, Gev ^ -1
        self.mu = mass_number * 0.938                                                           # nucleus (reduced) mass, GeV
        self.mass_DM = mass_DM                                                                  # DM mass, Gev
        self.photon_mass = 0                                                                    # photon mass, Gev

    def within_bounds(self, pt):
        """
        checks if a given point is within the bounds of the detector
        :param pt: point (TVector3) in space
        :return: 1 if within bounds, 0 if not
        """
        if (pt.x <= self.x_max) & (pt.x >= self.x_min) & (pt.y <= self.y_max) & (pt.y >= self.y_min) & (pt.z <= self.z_max) & (pt.z >= self.z_min):
            return 1
        else:
            return 0


class DynamicDarkMatter:
    def __init__(self, model):
        self.model = model
        self.event = hm.GenEvent()

        self.entry = None
        self.exit = None
        self.speed = None
        self.traj_vect_norm = None

        self.tot_dist = 0
        self.all_inter_pts = []
        self.curr_time = 0

        self.cross_section_total = rates.xsec_v_tot_S()

    def trajectory(self):
        """
        solves for a single sampled trajectory of the DM through the detector
        :return:
        entry - the entry point (TVector3) of the DM into the detector
        exit - the exit point (TVector3) of the DM out of the detector
        speed - the sampled speed of the DM through the detector
        traj_vect_norm - normalized trajectory vector of the DM
        """
        c = 2.0 * np.random.random() - 1.0
        theta = np.arccos(c)
        phi = 2.0 * np.pi * np.random.random()
        alpha = 2.0 * np.pi * np.random.random()

        prob_r = np.random.random()
        r_max = np.sqrt(((self.model.x_max * 2) ** 2) + ((self.model.y_max * 2) ** 2) + ((self.model.z_max * 2) ** 2)) / 2
        r = r_max * np.sqrt(prob_r)

        disk = urm.TVector3(np.cos(alpha), np.sin(alpha), 0).rotatey(theta).rotatez(phi) * r
        orient = [np.sqrt(1 - c ** 2) * np.cos(phi), np.sqrt(1 - c ** 2) * np.sin(phi), c]

        faces = []
        lambdas = np.zeros(6)
        valid_points = []
        valid_lambdas = []

        lambdas[0] = (self.model.z_max - disk.z) / orient[2]
        faces.append(urm.TVector3(lambdas[0] * orient[0] + disk.x, lambdas[0] * orient[1] + disk.y, self.model.z_max))
        lambdas[1] = (self.model.z_min - disk.z) / orient[2]
        faces.append(urm.TVector3(lambdas[1] * orient[0] + disk.x, lambdas[1] * orient[1] + disk.y, self.model.z_min))
        lambdas[2] = (self.model.x_max - disk.x) / orient[0]
        faces.append(urm.TVector3(self.model.x_max, lambdas[2] * orient[1] + disk.y, lambdas[2] * orient[2] + disk.z))
        lambdas[3] = (self.model.x_min - disk.x) / orient[0]
        faces.append(urm.TVector3(self.model.x_min, lambdas[3] * orient[1] + disk.y, lambdas[3] * orient[2] + disk.z))
        lambdas[4] = (self.model.y_max - disk.y) / orient[1]
        faces.append(urm.TVector3(lambdas[4] * orient[0] + disk.x, self.model.y_max, lambdas[4] * orient[2] + disk.z))
        lambdas[5] = (self.model.y_min - disk.y) / orient[1]
        faces.append(urm.TVector3(lambdas[5] * orient[0] + disk.x, self.model.y_min, lambdas[5] * orient[2] + disk.z))

        for i in range(6):
            if self.model.within_bounds(faces[i]):
                valid_points.append(faces[i])
                valid_lambdas.append(lambdas[i])

        if len(valid_points) == 2:
            if valid_lambdas[0] < valid_lambdas[1]:
                self.entry = valid_points[0]
                self.exit = valid_points[1]
            else:
                self.entry = valid_points[1]
                self.exit = valid_points[0]

            v_bar = (10 ** -3)
            v_x = np.random.normal(0, v_bar / np.sqrt(3))
            v_y = np.random.normal(0, v_bar / np.sqrt(3))
            v_z = np.random.normal(0, v_bar / np.sqrt(3))
            self.speed = np.sqrt((v_x ** 2) + (v_y ** 2) + (v_z ** 2))

            traj_vect = urm.TVector3(self.exit.x - self.entry.x, self.exit.y - self.entry.y, self.exit.z - self.entry.z)
            self.traj_vect_norm = traj_vect / np.sqrt((traj_vect.x ** 2) + (traj_vect.y ** 2) + (traj_vect.z ** 2))

            return

        elif len(valid_points) == 0:
            return self.trajectory()

        elif len(valid_points) == 1 or len(valid_points) > 2:
            raise ValueError('Invalid number of trajectory entry/exit points')

    def nextInteractionPoint(self):
        """
        samples the next interaction point along the trajectory and checks if it is within bounds of detector
        :return: inter_pt if within bounds of detector, 0 if not within bounds of detector
        """
        y = np.random.random()
        dist = -(1 / (self.model.number_density_target * self.cross_section_total)) * np.log(1 - y)
        self.tot_dist = self.tot_dist + dist

        inter_pt = self.entry + self.tot_dist * self.traj_vect_norm
        self.all_inter_pts.append(inter_pt)

        if len(self.all_inter_pts) > 1:
            net_time = dist / self.speed
            self.curr_time = self.curr_time + net_time

        if self.model.within_bounds(inter_pt):
            return inter_pt
        else:
            return 0

    def scattering(self):
        inter_pt = self.nextInteractionPoint()

        if inter_pt != 0:
            target = urm.TLorentzVector(0, 0, 0, self.model.molar_mass_target * 0.9315)     # Gev

            gamma = 1 / np.sqrt(1 - (self.speed ** 2))
            p_DM_in = gamma * self.speed * self.traj_vect_norm
            p_mag_DM_in = np.sqrt((p_DM_in.x ** 2) + (p_DM_in.y ** 2) + (p_DM_in.z ** 2))
            DM_in = urm.TLorentzVector(p_DM_in.x, p_DM_in.y, p_DM_in.z, np.sqrt((p_mag_DM_in ** 2) + self.model.mass_DM ** 2))

            boost_factor = -urm.TVector3(DM_in.x + target.x, DM_in.y + target.y, DM_in.z + target.z) / (DM_in.E + target.E)
            target_boosted = target.boost(boost_factor)
            beam_boosted = DM_in.boost(boost_factor)

            mass_out_DM = self.model.mass_DM + self.model.molar_mass_target - self.model.V0

            E_cm = beam_boosted.E + target_boosted.E
            E_out_photon = (E_cm ** 2 + self.model.photon_mass ** 2 - mass_out_DM ** 2) / (2 * E_cm)
            E_out_DM = (E_cm ** 2 + mass_out_DM ** 2 - self.model.photon_mass ** 2) / (2 * E_cm)

            p_photon_abs = np.sqrt(E_out_photon ** 2 - self.model.photon_mass ** 2)

            c_out = 2.0 * np.random.random() - 1.0
            phi_out = 2.0 * np.pi * np.random.random()

            out_photon_boosted = urm.TLorentzVector(p_photon_abs * np.sqrt(1 - c_out ** 2) * np.cos(phi_out),
                                                    p_photon_abs * np.sqrt(1 - c_out ** 2) * np.sin(phi_out),
                                                    p_photon_abs * c_out, E_out_photon)
            out_DM_boosted = urm.TLorentzVector(-p_photon_abs * np.sqrt(1 - c_out ** 2) * np.cos(phi_out),
                                                -p_photon_abs * np.sqrt(1 - c_out ** 2) * np.sin(phi_out),
                                                -p_photon_abs * c_out, E_out_DM)

            inverse_boost_factor = -boost_factor
            out_photon = out_photon_boosted.boost(inverse_boost_factor)
            out_DM = out_DM_boosted.boost(inverse_boost_factor)

            vertex_scatter = hm.GenVertex(hm.FourVector(self.curr_time, inter_pt.x, inter_pt.y, inter_pt.z))
            scattered_photon = hm.GenParticle(hm.FourVector(out_photon.E, out_photon.x, out_photon.y, out_photon.z), 22)
            vertex_scatter.add_particle_out(scattered_photon)
            self.event.add_vertex(vertex_scatter)

    def decay(self, inter_pt):
        decay_rate = rates.Gamma_tot_B()
        while decay_rate > 0:
            tr = np.random.random()
            decay_time = (-1 / decay_rate) * np.log(1 - tr)
            self.curr_time = self.curr_time + decay_time
            vertex_decay = hm.GenVertex(hm.FourVector(self.curr_time, inter_pt.x, inter_pt.y, inter_pt.z))

            c_decay = 2.0 * np.random.random() - 1.0
            phi_decay = 2.0 * np.pi * np.random.random()

            E_decay_photon = self.model.V0
            p_decay_photon = np.sqrt(E_decay_photon ** 2 - self.model.photon_mass ** 2)

            decay_photon = hm.GenParticle(hm.FourVector(E_decay_photon, p_decay_photon * np.sqrt(1 - c_decay ** 2)
                                                        * np.cos(phi_decay), p_decay_photon * np.sqrt(1 - c_decay ** 2)
                                                        * np.sin(phi_decay), p_decay_photon * c_decay), 22)
            vertex_decay.add_particle_out(decay_photon)
            self.event.add_vertex(vertex_decay)

    def Rates(self):
        # Derivatives of spherical bessel functions
        def spherical_jnp(l, x):
            return -0.5 * spherical_jn(l, x) / x + np.sqrt(0.5 * np.pi / x) * jvp(l + 0.5, x)

        def spherical_ynp(l, x):
            return -0.5 * spherical_yn(l, x) / x + np.sqrt(0.5 * np.pi / x) * yvp(l + 0.5, x)

        # Spherical bessel function zeros as floats (no mpmath)
        def spherical_jnz(l, n):
            return float(besseljzero(l + 0.5, n))

        ################################################################################
        # PARAMETERS
        ################################################################################


        k = 1.0e-3  # incoming DM momentum, GeV
        levels = {}  # cache for energy levels

        ################################################################################
        # STATES
        ################################################################################

        kapS = np.sqrt(k ** 2 + 2.0 * self.model.mu * self.model.V0)  # interior momentum for scattering, GeV

        # momentum inside potential well for bound state, GeV
        def kapB(n, l):
            return spherical_jnz(l, n) / self.model.R_phi

        # (positive) binding energy for state, GeV
        def EB(n, l):
            try:
                return levels[(n, l)]
            except KeyError:
                res = self.model.V0 - 0.5 * kapB(n, l) ** 2 / self.model.mu
                if res > 0.:
                    levels[(n, l)] = res
                return res

        # emitted photon energy/momentum, GeV
        def q(ni, li, nf, lf):
            if - EB(ni, li) + EB(nf, lf) <= 0.:
                raise ValueError('Attempting transition from lower energy state to higher energy state')
            return - EB(ni, li) + EB(nf, lf)

        # Maximum n allowed to have bound states below top of potential
        def nmax(l, Emin):
            res = int(np.ceil((np.sqrt(2.0 * self.model.mu * self.model.V0) * self.model.R_phi) / np.pi - 0.5 * l))
            while EB(res, l) < Emin:
                res -= 1
            while EB(res + 1, l) > Emin:
                res += 1
            return res

        # normalization for bound state, GeV^-3/2
        def NB(n, l):
            normint = -0.25 * (np.pi * self.model.R_phi ** 3 * jv(-0.5 + l, spherical_jnz(l, n)) * jv(1.5 + l, spherical_jnz(l,n))) / spherical_jnz(l, n)
            return 1.0 / np.sqrt(normint)

        # boundary conditions for scattering state
        def bcs(Ns, delta, l):
            return (
            (np.cos(delta) * spherical_jn(l, k * self.model.R_phi) + np.sin(delta) * spherical_yn(l, k * self.model.R_phi)) - Ns * spherical_jn(l, kapS * self.model.R_phi),
            (np.cos(delta) * k * spherical_jnp(l, k * self.model.R_phi) + np.sin(delta) * k * spherical_ynp(l, k * self.model.R_phi)) - Ns * kapS * spherical_jnp(l, kapS * self.model.R_phi))

        # solve boundary conditions to get interior normalization, dimensionless
        def NS(l):
            return 4.0 * np.pi * 1j ** l * fsolve(lambda x: bcs(x[0], x[1], l), (1.0, 0.3))[0]

        # interior wave function profiles for scattering and bound states, dimensionless
        def RS(r, l):
            return spherical_jn(l, kapS * r)

        def RB(r, n, l):
            return spherical_jn(l, kapB(n, l) * r)

        ################################################################################
        # INTEGRALS
        ################################################################################

        # radial intergral for decay in dipole approximation
        # dimension GeV^-4
        # cache the results to save time
        # added approximate result at large kappa R
        rad_int_B_cache = {}

        def rad_int_B(ni, li, nf, lf, force_full=False, subinterval_periods=8.0, approx_threshold=10.0):
            if abs(li - lf) != 1:
                raise ValueError('Calculating amplitude for unallowed transition')
            try:
                res = rad_int_B_cache[(ni, li, nf, lf)]
            except KeyError:
                subinterval_lim = max(50, int(np.ceil(max(kapS * self.model.R_phi, kapB(nf, lf) * self.model.R_phi) / subinterval_periods)))
                if kapB(ni, li) * self.model.R_phi < approx_threshold or kapB(nf, lf) * self.model.R_phi < approx_threshold or force_full:
                    rad_int_full = quad(lambda r: RB(r, ni, li) * RB(r, nf, lf) * r ** 3, 0, self.model.R_phi, limit=subinterval_lim)
                    if rad_int_full[1] > 0.01 * abs(rad_int_full[0]):
                        print('Error on radial integral greater than 1%')
                    res = rad_int_full[0]
                else:
                    kap1 = kapB(ni, li)
                    kap2 = kapB(nf, lf)
                    kap_sum = kap1 + kap2
                    kap_dif = kap2 - kap1
                    kap_sum_R = kap_sum * self.model.R_phi
                    kap_dif_R = kap_dif * self.model.R_phi
                    res = kap_dif ** 2 * (-1) ** ((1 + lf + li) / 2) * (
                                kap_sum_R * np.cos(kap_sum_R) - np.sin(kap_sum_R))
                    res += kap_sum ** 2 * (-1) ** ((1 + lf - li) / 2) * (
                                kap_dif_R * np.cos(kap_dif_R) - np.sin(kap_dif_R))
                    res /= 2.0 * kap1 * kap2 * kap_sum ** 2 * kap_dif ** 2
                rad_int_B_cache[(ni, li, nf, lf)] = res
            return res

        # radial intergral for scattering in dipole approximation
        # dimension GeV^-4
        rad_int_S_cache = {}

        def rad_int_S(li, nf, lf, force_full=False, subinterval_periods=8.0, approx_threshold=10.0):
            if abs(li - lf) != 1:
                raise ValueError('Calculating amplitude for unallowed transition')
            try:
                res = rad_int_S_cache[(li, nf, lf)]
            except KeyError:
                subinterval_lim = max(50, int(np.ceil(max(kapS * self.model.R_phi, kapB(nf, lf) * self.model.R_phi) / subinterval_periods)))
                if kapS * self.model.R_phi < approx_threshold or kapB(nf, lf) * self.model.R_phi < approx_threshold or force_full:
                    rad_int_full = quad(lambda r: RS(r, li) * RB(r, nf, lf) * r ** 3, 0, self.model.R_phi, limit=subinterval_lim)
                    if rad_int_full[1] > 0.01 * abs(rad_int_full[0]):
                        print('Error on radial integral greater than 1%')
                    res = rad_int_full[0]
                else:
                    kapB_cur = kapB(nf, lf)
                    kap_sum = kapB_cur + kapS
                    kap_dif = kapB_cur - kapS
                    kap_sum_R = kap_sum * self.model.R_phi
                    kap_dif_R = kap_dif * self.model.R_phi
                    res = kap_dif ** 2 * (-1) ** ((1 + lf + li) / 2) * (
                                kap_sum_R * np.cos(kap_sum_R) - np.sin(kap_sum_R))
                    res += kap_sum ** 2 * (-1) ** ((1 + lf - li) / 2) * (
                                kap_dif_R * np.cos(kap_dif_R) - np.sin(kap_dif_R))
                    res /= 2.0 * kapB_cur * kapS * kap_sum ** 2 * kap_dif ** 2
                rad_int_S_cache[(li, nf, lf)] = res
            return res

        # Triple product of spherical harmonics, integrated
        # This combination appears in the angular integrals
        def sph_prod(li, mi, mr, lf, mf):
            if abs(li - lf) != 1 or mi + mr != mf:
                return 0.0
            coef = np.sqrt(3.0 / (4.0 * np.pi * (2 * lf + 1)))
            clres = 0.
            if lf == li + 1:
                clres = np.sqrt(comb(li + 1 - mi - mr, 1 - mr) * comb(li + 1 + mi + mr, 1 + mr))
            elif lf == li - 1:
                clres = np.sqrt(comb(li + mi, 1 - mr) * comb(li - mi, 1 + mr))
            else:
                raise ValueError('Unphysical spherical harmonic product')
            return coef * clres

        # Angular integral assembled for all three components
        def ang_int(li, mi, lf, mf):
            if abs(li - lf) != 1 or abs(mi - mf) > 1:
                return 0.0
            sph_x = sph_prod(li, mi, -1, lf, mf) - sph_prod(li, mi, 1, lf, mf)
            sph_y = 1j * (sph_prod(li, mi, -1, lf, mf) - sph_prod(li, mi, 1, lf, mf))
            sph_z = np.sqrt(2.0) * sph_prod(li, mi, 0, lf, mf)
            return np.sqrt(2.0 * np.pi / 3.0) * np.array([sph_x, sph_y, sph_z])

        # amplitude
        # dimesionless
        def amp_B(ni, li, mi, nf, lf, mf, force_full=False, subinterval_periods=8.0, approx_threshold=10.0):
            return self.model.atomic_number_target * self.model.electric_charge * q(ni, li, nf, lf) * NB(ni, li) * NB(nf, lf) * rad_int_B(ni, li, nf, lf, force_full, subinterval_periods, approx_threshold) * ang_int(li, mi, lf, mf)

        # amplitude for scattering
        # in GeV^-3/2
        def amp_S(nf, lf, mf, force_full=False, subinterval_periods=8.0, approx_threshold=10.0):
            res = np.array([0., 0., 0.], dtype='complex128')
            for li in [lf - 1, lf + 1]:
                if li < 0 or k * self.model.R_phi < li:
                    continue
                res += self.model.atomic_number_target * self.model.electric_charge * (EB(nf, lf) + k ** 2 / (2.0 * self.model.mu)) * NS(li) * NB(nf, lf) * np.sqrt(
                    (2.0 * li + 1) / (4.0 * np.pi)) * rad_int_S(li, nf, lf, force_full, subinterval_periods,
                                                                approx_threshold) * ang_int(li, 0, lf, mf)
            return res

        ################################################################################
        # MAIN DECAY FUNCTIONS
        ################################################################################

        # decay rate differential in cos(theta) for the photon (ctq) relative to spin z axis, GeV
        # eps: polarization of outgoing photon (+- 1 for right/left)
        # ni,li,mi: initial state quantum numbers
        # nf,lf,mf: final state quantum numbers
        def dGamma_B(ctq, ni, li, mi, nf, lf, mf, force_full=False, subinterval_periods=8.0, approx_threshold=10.0):
            stq = np.sqrt(1.0 - ctq ** 2)
            pol_mat = np.array([[ctq ** 2, 0., -ctq * stq], [0., 1., 0.], [-ctq * stq, 0., stq ** 2]])
            amp = amp_B(ni, li, mi, nf, lf, mf, force_full, subinterval_periods, approx_threshold)
            return np.real(q(ni, li, nf, lf) * np.linalg.multi_dot([np.conjugate(amp), pol_mat, amp]) / (16.0 * np.pi))

        # total decay rate in GeV
        # Independent of polarization (emerges as phase)
        # ni,li,mi: initial state quantum numbers
        # nf,lf,mf: final state quantum numbers
        def Gamma_B(ni, li, mi, nf, lf, mf, force_full=False, subinterval_periods=8.0, approx_threshold=10.0):
            pol_mat = np.array([[2.0 / 3, 0., 0.], [0., 2.0, 0.], [0., 0., 4.0 / 3.]])
            amp = amp_B(ni, li, mi, nf, lf, mf, force_full, subinterval_periods, approx_threshold)
            return np.real(q(ni, li, nf, lf) * np.linalg.multi_dot([np.conjugate(amp), pol_mat, amp]) / (
                        16.0 * np.pi))  # decay rate to all allowed states in GeV

        # decay rate to all allowed states in GeV
        # n,l,m: quantum numbers of decaying state
        # Returns: all allowed n,l,m final states with their respective decay rate
        def Gamma_tot_B(n, l, m, force_full=False, subinterval_periods=8.0, approx_threshold=10.0):
            res = {}
            for lf in [l - 1, l + 1]:
                if lf < 0:
                    continue
                nf = nmax(lf, EB(n, l))
                while nf > 0 and q(n, l, nf, lf) * self.model.R_phi < np.pi:
                    for mf in range(m - 1, m + 1):
                        if mf > lf or mf < -lf:
                            continue
                        rate = Gamma_B(n, l, m, nf, lf, mf, force_full, subinterval_periods, approx_threshold)
                        res[(nf, lf, mf)] = rate
                    nf -= 1
            return res

        ################################################################################
        # MAIN SCATTERING FUNCTIONS
        ################################################################################

        # Main scattering functions
        # scattering rate differential in cos(theta) for the photon (ctq) relative to spin z axis, GeV^-2
        def dxsec_v_S(ctq, nf, lf, mf, force_full=False, subinterval_periods=8.0, approx_threshold=10.0):
            stq = np.sqrt(1.0 - ctq ** 2)
            pol_mat = np.array([[ctq ** 2, 0., -ctq * stq], [0., 1., 0.], [-ctq * stq, 0., stq ** 2]])
            amp = amp_S(nf, lf, mf, force_full, subinterval_periods, approx_threshold)
            return np.real(EB(nf, lf) * np.linalg.multi_dot([np.conjugate(amp), pol_mat, amp]) / (4.0 * np.pi))

        # total cross-section to given final state in GeV^-2
        def xsec_v_S(nf, lf, mf, force_full=False, subinterval_periods=8.0, approx_threshold=10.0):
            pol_mat = np.array([[2.0 / 3, 0., 0.], [0., 2.0, 0.], [0., 0., 4.0 / 3.]])
            amp = amp_S(nf, lf, mf, force_full, subinterval_periods, approx_threshold)
            return np.real(EB(nf, lf) * np.linalg.multi_dot([np.conjugate(amp), pol_mat, amp]) / (4.0 * np.pi))

        # cross-section to all allowed states in GeV^-2
        def xsec_v_tot_S(force_full=False, subinterval_periods=8.0, approx_threshold=10.0):
            res = {}
            for lf in range(int(np.ceil(k * self.model.R_phi)) + 1):
                nf = nmax(lf, 0.)
                for mf in range(-1, 2):
                    if mf < -lf or mf > lf:
                        continue
                    xsec_v = xsec_v_S(nf, lf, mf, force_full, subinterval_periods, approx_threshold)
                    res[(nf, lf, mf)] = xsec_v
            return res


class Simulation:
    def __init__(self, model, iterations):
        self.traj_events = None
        self.model = model
        self.iterations = iterations

    def run(self):
        # arrays to store trajectory events
        traj_events_all = np.array([])

        # iterates the scattering_interation method of the model for given number of trajectory attempts
        for n in range(self.iterations):
            # stores photons from specific trajectory
            traj_event = self.model.scattering_interaction()
            np.append(traj_events_all, traj_event)

        # stores filled array of photons emitted as property of the class
        self.traj_events = traj_events_all

    def write(self):
        # creates file to store events to
        writer = hm.io.WriterAscii('photons_emitted.HepMC3')

        # iterates through all trajectories and all events writing each to the writer
        for i in range(len(self.traj_events)):
            writer.write_event(self.traj_events[i])

    def read_momenta(self, filename):
        reader = hm.io.ReaderAscii(filename)
        momenta = np.array([])
        traj_event = hm.GenEvent()
        while traj_event is not None:
            reader.read_event(traj_event)
            for vertex in traj_event.vertices:
                for particle in vertex.particles_out:
                    np.append(momenta, particle.momentum)

        return momenta
