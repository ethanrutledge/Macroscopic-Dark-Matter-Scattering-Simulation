################################################################################
# ENVIRONMENT
################################################################################

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

# Derivatives of spherical bessel functions
def spherical_jnp(l,x):
    return -0.5 * spherical_jn(l,x) / x + np.sqrt(0.5 * np.pi / x) * jvp(l+0.5,x)
def spherical_ynp(l,x):
    return -0.5 * spherical_yn(l,x) / x + np.sqrt(0.5 * np.pi / x) * yvp(l+0.5,x)
# Spherical bessel function zeros as floats (no mpmath)
def spherical_jnz(l,n):
    return float(besseljzero(l+0.5,n))

################################################################################
# PARAMETERS
################################################################################

Z = 18                                 # material atomic number
A = 40                                 # material mass number
e = np.sqrt(4.0 * np.pi / 137)         # electric charge, dimensionless
mu = A * 0.938                         # nucleus (reduced) mass, GeV
V0 = A * 0.246                         # potential depth, GeV
k = 1.0e-3 * mu                        # incoming DM momentum, GeV
R = 10.0                               # DM radius, GeV^-1
levels = {}                            # cache for energy levels

################################################################################
# STATES
################################################################################

kapS = np.sqrt(k**2 + 2.0 * mu * V0)   # interior momentum for scattering, GeV

# momentum inside potential well for bound state, GeV
def kapB(n,l):
    return spherical_jnz(l,n) / R
# (positive) binding energy for state, GeV
def EB(n,l):
    try:
        return levels[(n,l)]
    except KeyError:
        res = V0 - 0.5 * kapB(n,l)**2 / mu
        if res > 0.:
            levels[(n,l)] = res
        return res
# emitted photon energy/momentum, GeV
def q(ni,li,nf,lf):
    if - EB(ni,li) + EB(nf,lf) <= 0.:
        raise ValueError('Attempting transition from lower energy state to higher energy state')
    return - EB(ni,li) + EB(nf,lf)

# Maximum n allowed to have bound states below top of potential
def nmax(l,Emin):
    res = int(np.ceil((np.sqrt(2.0*mu*V0)*R)/np.pi - 0.5 * l))
    while res > 0 and EB(res,l) < Emin:
        res -= 1
    while res + 1 > 0 and EB(res+1,l) > Emin:
        res += 1
    return res

# normalization for bound state, GeV^-3/2
def NB(n,l):
    normint = -0.25*(np.pi*R**3*jv(-0.5 + l,spherical_jnz(l,n))*jv(1.5 + l,spherical_jnz(l,n)))/spherical_jnz(l,n)
    return 1.0/np.sqrt(normint)
# boundary conditions for scattering state
def bcs(Ns,delta,l):
    return ((np.cos(delta) * spherical_jn(l,k*R) + np.sin(delta) * spherical_yn(l,k*R)) - Ns * spherical_jn(l,kapS*R),
            (np.cos(delta) * k * spherical_jnp(l,k*R) + np.sin(delta) * k * spherical_ynp(l,k*R)) - Ns * kapS * spherical_jnp(l,kapS*R))
# solve boundary conditions to get interior normalization, dimensionless
def NS(l):
    return 4.0*np.pi*1j**l * fsolve(lambda x : bcs(x[0],x[1],l),(1.0,0.3))[0]

# interior wave function profiles for scattering and bound states, dimensionless
def RS(r,l):
    return spherical_jn(l,kapS*r)
def RB(r,n,l):
    return spherical_jn(l,kapB(n,l)*r)

################################################################################
# INTEGRALS
################################################################################

# radial intergral for decay in dipole approximation
# dimension GeV^-4
# cache the results to save time
# added approximate result at large kappa R
rad_int_B_cache = {}
def rad_int_B(ni,li,nf,lf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    if abs(li-lf) != 1:
        raise ValueError('Calculating amplitude for unallowed transition')
    try:
        res = rad_int_B_cache[(ni,li,nf,lf)]
    except KeyError:
        subinterval_lim = max(50,int(np.ceil(max(kapS*R,kapB(nf,lf)*R)/subinterval_periods)))
        if kapB(ni,li)*R < approx_threshold or kapB(nf,lf)*R < approx_threshold or force_full:
            rad_int_full = quad(lambda r : RB(r,ni,li)*RB(r,nf,lf)*r**3,0,R,limit=subinterval_lim)
            if rad_int_full[1] > 0.01 * abs(rad_int_full[0]):
                print('Error on radial integral greater than 1%')
            res = rad_int_full[0]
        else:
            kap1 = kapB(ni,li)
            kap2 = kapB(nf,lf)
            kap_sum = kap1 + kap2
            kap_dif = kap2 - kap1
            kap_sum_R = kap_sum*R
            kap_dif_R = kap_dif*R
            res =  kap_dif**2 * (-1)**((1+lf+li)/2) * (kap_sum_R*np.cos(kap_sum_R) - np.sin(kap_sum_R))
            res += kap_sum**2 * (-1)**((1+lf-li)/2) * (kap_dif_R*np.cos(kap_dif_R) - np.sin(kap_dif_R))
            res /= 2.0 * kap1 * kap2 * kap_sum**2 * kap_dif**2
        rad_int_B_cache[(ni,li,nf,lf)] = res
    return res
# radial intergral for scattering in dipole approximation
# dimension GeV^-4
rad_int_S_cache = {}
def rad_int_S(li,nf,lf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    if abs(li-lf) != 1:
        raise ValueError('Calculating amplitude for unallowed transition')
    try:
        res = rad_int_S_cache[(li,nf,lf)]
    except KeyError:
        subinterval_lim = max(50,int(np.ceil(max(kapS*R,kapB(nf,lf)*R)/subinterval_periods)))
        if kapS*R < approx_threshold or kapB(nf,lf)*R < approx_threshold or force_full:
            rad_int_full = quad(lambda r : RS(r,li)*RB(r,nf,lf)*r**3,0,R,limit=subinterval_lim)
            if rad_int_full[1] > 0.01 * abs(rad_int_full[0]):
                print('Error on radial integral greater than 1%')
            res = rad_int_full[0]
        else:
            kapB_cur = kapB(nf,lf)
            kap_sum = kapB_cur + kapS
            kap_dif = kapB_cur - kapS
            kap_sum_R = kap_sum*R
            kap_dif_R = kap_dif*R
            res =  kap_dif**2 * (-1)**((1+lf+li)/2) * (kap_sum_R*np.cos(kap_sum_R) - np.sin(kap_sum_R))
            res += kap_sum**2 * (-1)**((1+lf-li)/2) * (kap_dif_R*np.cos(kap_dif_R) - np.sin(kap_dif_R))
            res /= 2.0 * kapB_cur * kapS * kap_sum**2 * kap_dif**2
        rad_int_S_cache[(li,nf,lf)] = res
    return res

# Triple product of spherical harmonics, integrated
# This combination appears in the angular integrals
def sph_prod(li,mi,mr,lf,mf):
    if abs(li-lf) != 1 or mi+mr != mf:
        return 0.0
    coef = np.sqrt(3.0/(4.0 * np.pi * (2*lf+1)))
    clres = 0.
    if lf == li + 1:
        clres = np.sqrt(comb(li+1-mi-mr,1-mr)*comb(li+1+mi+mr,1+mr))
    elif lf == li - 1:
        clres = np.sqrt(comb(li+mi,1-mr)*comb(li-mi,1+mr))
    else:
        raise ValueError('Unphysical spherical harmonic product')
    return coef * clres

# Angular integral assembled for all three components
def ang_int(li,mi,lf,mf):
    if abs(li-lf) != 1 or abs(mi-mf) > 1:
        return 0.0
    sph_x = sph_prod(li,mi,-1,lf,mf)-sph_prod(li,mi,1,lf,mf)
    sph_y = 1j*(sph_prod(li,mi,-1,lf,mf)-sph_prod(li,mi,1,lf,mf))
    sph_z = np.sqrt(2.0) * sph_prod(li,mi,0,lf,mf)
    return np.sqrt(2.0*np.pi/3.0) * np.array([sph_x,sph_y,sph_z])

# amplitude
# dimesionless
def amp_B(ni,li,mi,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    return Z * e * q(ni,li,nf,lf) * NB(ni,li) * NB(nf,lf) * rad_int_B(ni,li,nf,lf,force_full,subinterval_periods,approx_threshold) * ang_int(li,mi,lf,mf)

# amplitude for scattering
# in GeV^-3/2
def amp_S(nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    res = np.array([0.,0.,0.],dtype='complex128')
    for li in [lf-1,lf+1]:
        if li < 0 or k*R < li:
            continue
        res += Z * e * (EB(nf,lf) + k**2/(2.0 * mu)) * NS(li) * NB(nf,lf) * np.sqrt((2.0*li+1)/(4.0*np.pi)) * rad_int_S(li,nf,lf,force_full,subinterval_periods,approx_threshold) * ang_int(li,0,lf,mf)
    return res

################################################################################
# MAIN DECAY FUNCTIONS
################################################################################

# decay rate differential in cos(theta) for the photon (ctq) relative to spin z axis, GeV
# eps: polarization of outgoing photon (+- 1 for right/left)
# ni,li,mi: initial state quantum numbers
# nf,lf,mf: final state quantum numbers
def dGamma_B(ctq,ni,li,mi,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    stq = np.sqrt(1.0-ctq**2)
    pol_mat = np.array([[ctq**2,0.,-ctq*stq],[0.,1.,0.],[-ctq*stq,0.,stq**2]])
    amp = amp_B(ni,li,mi,nf,lf,mf,force_full,subinterval_periods,approx_threshold)
    return np.real(q(ni,li,nf,lf) * np.linalg.multi_dot([np.conjugate(amp),pol_mat,amp]) / (16.0 * np.pi))

# total decay rate in GeV
# Independent of polarization (emerges as phase)
# ni,li,mi: initial state quantum numbers
# nf,lf,mf: final state quantum numbers
def Gamma_B(ni,li,mi,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    pol_mat = np.array([[2.0/3,0.,0.],[0.,2.0,0.],[0.,0.,4.0/3.]])
    amp = amp_B(ni,li,mi,nf,lf,mf,force_full,subinterval_periods,approx_threshold)
    return np.real(q(ni,li,nf,lf) * np.linalg.multi_dot([np.conjugate(amp),pol_mat,amp]) / (16.0 * np.pi)) # decay rate to all allowed states in GeV

# decay rate to all allowed states in GeV
# n,l,m: quantum numbers of decaying state
# Returns: all allowed n,l,m final states with their respective decay rate
def Gamma_tot_B(n,l,m,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    res = {}
    for lf in [l-1,l+1]:
        if lf < 0:
            continue
        nf = nmax(lf,EB(n,l))
        while nf > 0 and q(n,l,nf,lf)*R < np.pi:
            for mf in range(m-1,m+1):
                if mf > lf or mf < -lf:
                    continue
                rate = Gamma_B(n,l,m,nf,lf,mf,force_full,subinterval_periods,approx_threshold)
                res[(nf,lf,mf)] = rate
            nf -= 1
    return res

################################################################################
# MAIN SCATTERING FUNCTIONS
################################################################################

# Main scattering functions
# scattering rate differential in cos(theta) for the photon (ctq) relative to spin z axis, GeV^-2
def dxsec_v_S(ctq,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    stq = np.sqrt(1.0-ctq**2)
    pol_mat = np.array([[ctq**2,0.,-ctq*stq],[0.,1.,0.],[-ctq*stq,0.,stq**2]])
    amp = amp_S(nf,lf,mf,force_full,subinterval_periods,approx_threshold)
    return np.real(EB(nf,lf) * np.linalg.multi_dot([np.conjugate(amp),pol_mat,amp]) / (4.0 * np.pi))

# total cross section to given final state in GeV^-2
def xsec_v_S(nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    pol_mat = np.array([[2.0/3,0.,0.],[0.,2.0,0.],[0.,0.,4.0/3.]])
    amp = amp_S(nf,lf,mf,force_full,subinterval_periods,approx_threshold)
    return np.real(EB(nf,lf) * np.linalg.multi_dot([np.conjugate(amp),pol_mat,amp]) / (4.0 * np.pi))

# cross section to all allowed states in GeV^-2
def xsec_v_tot_S(force_full = False, subinterval_periods = 8.0, approx_threshold = 10.0):
    res = {}
    for lf in range(int(np.ceil(k*R)) + 1):
        nf = nmax(lf,0.)
        for mf in range(-1,2):
            if mf < -lf or mf > lf:
                continue
            xsec_v = xsec_v_S(nf,lf,mf,force_full,subinterval_periods,approx_threshold)
            res[(nf,lf,mf)] = xsec_v
    return res
