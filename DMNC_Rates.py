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

################################################################################
# STATES
################################################################################

kapS = np.sqrt(k**2 + 2.0 * mu * V0)   # interior momentum for scattering, GeV

# momentum inside potential well for bound state, GeV
def kapB(n,l):
    return spherical_jnz(l,n) / R
# (positive) binding energy for state, GeV
def EB(n,l):
    return V0 - 0.5 * kapB(n,l)**2 / mu
# emitted photon energy/momentum, GeV
def q(ni,li,nf,lf):
    if - EB(ni,li) + EB(nf,lf) <= 0.:
        raise ValueError('Attempting transition from lower energy state to higher energy state')
    return - EB(ni,li) + EB(nf,lf)

# calculate all allowed energy levels, GeV
# this might be very slow for large radii--any way to speed up?
# high order Bessel are very slow
levels = []
ncur = 1
lcur = 0
while EB(ncur,lcur) > 0.0:
    while EB(ncur,lcur) > 0.0:
        levels.append([ncur,lcur,EB(ncur,lcur)])
        lcur = lcur + 1
    lcur = 0
    ncur = ncur + 1
levels = sorted(levels,key = lambda x : x[2])

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
rad_int_B_cache = {}
def rad_int_B(ni,li,nf,lf):
    if abs(li-lf) != 1:
        raise ValueError('Calculating amplitude for unallowed transition')
    try:
        res = rad_int_B_cache[(ni,li,nf,lf)]
    except KeyError:
        rad_int_full = quad(lambda r : RB(r,ni,li) * RB(r,nf,lf) * r**3,0,R)
        if rad_int_full[1] > 0.01*abs(rad_int_full[0]):
            print('Error on radial integral greater than 1%')
        res = rad_int_full[0]
        rad_int_B_cache[(ni,li,nf,lf)] = res
    return res
# radial intergral for scattering in dipole approximation
# dimension GeV^-4
rad_int_S_cache = {}
def rad_int_S(li,nf,lf):
    if abs(li-lf) != 1:
        raise ValueError('Calculating amplitude for unallowed transition')
    try:
        res = rad_int_S_cache[(li,nf,lf)]
    except KeyError:
        rad_int_full = quad(lambda r : RS(r,li)*RB(r,nf,lf)*r**3,0,R)
        if rad_int_full[1] > 0.01 * abs(rad_int_full[0]):
            print('Error on radial integral greater than 1%')
        res = rad_int_full[0]
        rad_int_S_cache[(li,nf,lf)] = res
    return res
# angular intergral for allowed dipole transitions
# dimensionless
def ang_int(ctq,eps,li,mi,lf,mf):
    if lf == li - 1 and mf == mi - 1:
        return 0.5 * (eps - ctq) * np.sqrt((li+mi) * (li+mi-1)/(8.0 * li**2 - 2.0))
    elif lf == li - 1 and mf == mi:
        return - np.sqrt((1.0-ctq**2) * (li**2 - mi**2) / (8.0 * li**2 -2.0))
    elif lf == li - 1 and mf == mi + 1:
        return 0.5 * (ctq + eps) * np.sqrt((li-mi) * (li-mi-1)/(8.0 * li**2 - 2.0))
    elif lf == li + 1 and mf == mi - 1:
        return - 0.5 * (eps - ctq) * np.sqrt((li-mi+1) * (li-mi+2)/(8.0 * li**2 + 16.0 * li + 6.0))
    elif lf == li + 1 and mf == mi:
        return - np.sqrt((1.0-ctq**2) * (li-mi+1) * (li+mi+1)/(8.0 * li**2 + 16.0 * li + 6.0))
    elif lf == li + 1 and mf == mi + 1:
        return - 0.5 * (ctq + eps) * np.sqrt((li+mi+1) * (li+mi+2)/(8.0 * li**2 + 16.0 * li + 6.0))
    else:
        return 0.0

# angular integral of squared amplitude over photon direction
# dimensionless
# note: independent of photon polarization
def ang_int2_tot(li,mi,lf,mf):
    if lf == li - 1 and mf == mi - 1:
        return (-1. + li + mi)*(li + mi)/(-3. + 12.*li**2)
    elif lf == li - 1 and mf == mi:
        return (2.*(li - 1.*mi)*(li + mi))/(-3. + 12.*li**2)
    elif lf == li - 1 and mf == mi + 1:
        return ((-1. + li - 1.*mi)*(li - 1.*mi))/(-3. + 12.*li**2)
    elif lf == li + 1 and mf == mi - 1:
        return ((1. + li - 1.*mi)*(2. + li - 1.*mi))/(9. + 12.*li*(2. + li))
    elif lf == li + 1 and mf == mi:
        return (2.*(1. + li - 1.*mi)*(1. + li + mi))/(9. + 12.*li*(2. + li))
    elif lf == li + 1 and mf == mi + 1:
        return ((1. + li + mi)*(2. + li + mi))/(9. + 12.*li*(2. + li))
    else:
        return 0.0

# amplitude
# dimesionless
def amp_B(ctq,eps,ni,li,mi,nf,lf,mf):
    return Z * e * q(ni,li,nf,lf) * NB(ni,li) * NB(nf,lf) * rad_int(ni,li,nf,lf) * ang_int(ctq,eps,li,mi,lf,mf)

# amplitude for scattering
# in GeV^-3/2
def amp_S(ctq,eps,nf,lf,mf):
    res = 0.
    for li in [lf-1,lf+1]:
        if li < 0 or k*R < li:
            continue
        if k*R < li:
            continue
        for mi in range(-li,li+1):
            res += Z * e * (EB(nf,lf) + k**2/(2.0 * mu)) * NS(li) * NB(nf,lf) * rad_int_S(li,nf,lf) * ang_int(ctq,eps,li,mi,lf,mf)
    return res

################################################################################
# MAIN DECAY FUNCTIONS
################################################################################

# decay rate differential in cos(theta) for the photon (ctq) relative to spin z axis, GeV
# eps: polarization of outgoing photon (+- 1 for right/left)
# ni,li,mi: initial state quantum numbers
# nf,lf,mf: final state quantum numbers
def dGamma_B(ctq,eps,ni,li,mi,nf,lf,mf):
    return q(ni,li,nf,lf) * abs(amp_B(ctq,eps,ni,li,mi,nf,lf,mf))**2 / (16.0 * np.pi)

# total decay rate in GeV
# Independent of polarization (emerges as phase)
# ni,li,mi: initial state quantum numbers
# nf,lf,mf: final state quantum numbers
def Gamma_B(ni,li,mi,nf,lf,mf):
    return q(ni,li,nf,lf) * abs(Z * e * q(ni,li,nf,lf) * NB(ni,li) * NB(nf,lf) *  rad_int_B(ni,li,nf,lf))**2 * ang_int2_tot(li,mi,lf,mf) / (8.0 * np.pi)

# allowed final states for decay of state n,l,m
def allowed_fs(n,l,m):
    res = []
    for level in levels:
        nf = level[0]
        lf = level[1]
        # Only one unit of angular momentum change in dipole approx.
        if abs(lf - l) != 1:
            continue
        # Must go to lower energy state
        if - EB(n,l) + EB(nf,lf) <= 0:
            continue
        # Dipole approximation must be valid
        if q(n,l,nf,lf)*R > np.pi:
            continue
        res.append([nf,lf])
    return res

# decay rate to all allowed states in GeV
# n,l,m: quantum numbers of decaying state
# Returns: all allowed n,l,m final states with their respective decay rate
def Gamma_tot_B(n,l,m):
    res = []
    for level in allowed_fs(n,l,m):
        #print('Working on ',level[0],level[1])
        nf = level[0]
        lf = level[1]
        # Can only be m-1,m,m+1
        for mf in range(m-1,m+1):
            # For the m's: they must be in the correct range
            if mf > lf or mf < -lf:
                continue
            rate = Gamma_B(n,l,m,nf,lf,mf)
            res.append([nf,lf,mf,rate])
    return res

################################################################################
# MAIN SCATTERING FUNCTIONS
################################################################################

# Main scattering functions
# scattering rate differential in cos(theta) for the photon (ctq) relative to spin z axis, GeV^-2
def dxsec_v_S(ctq,eps,nf,lf,mf):
    return EB(nf,lf) * abs(amp_S(ctq,eps,nf,lf,mf))**2 / (4.0 * np.pi)
# total cross section to given final state in GeV^-2
rad_int_cache = {}
def xsec_v_S(nf,lf,mf):
    res_list = [quad(lambda ctq : dxsec_v_S(ctq,eps,nf,lf,mf),-1.0,1.0) for eps in [-1,1]]
    for res in res_list:
        if res[1] > 0.01 * abs(res[0]):
            print('Error on angular integral > 1%')
    res =  sum([r[0] for r in res_list])
    return res
# allowed final states for scattering
def allowed_fs_S():
    res = []
    for level in levels:
        nf = level[0]
        lf = level[1]
        # First condition: dipole approximation is valid
        # Second approximation: accessible dipole scattering channel for unsuppressed
        # scattering wavefunction angular momentum mode
        if level[2]*R > np.pi or k*R < lf-1:
            continue
        res.append([nf,lf])
    return res
# cross section to all allowed states in GeV^-2
def xsec_v_tot_S():
    res = []
    for level in allowed_fs_S():
        nf = level[0]
        lf = level[1]
        print('Working on ',level[0],level[1])
        for mf in range(-lf,lf+1):
            xsec_v = xsec_v_S(nf,lf,mf)
            res.append([nf,lf,mf,xsec_v])
    return res
