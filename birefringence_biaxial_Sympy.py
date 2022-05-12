#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:44:47 2020

@author: kyuubi
"""

"""
Same as biaxial 2 but using just sympy for testing

"""

import numpy as np
import mpmath
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sympy as sp
import sympy.plotting as sympl
import pandas as pd

plt.close('all')
fontsize = 14
plt.rcParams["font.size"] = fontsize
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

p = 20

#%%
# =============================================================================
# Functions
# =============================================================================

def Eend(E_inc, E_inc_s0, E_inc_p0, E_tf_t, E_ts_t, n_low, n_high, l_s, l_f, S_ts, S_tf, k_tr_f, k_tr_s, t, phase_jump):
    E_t_f_time = sp.re(E_inc.norm() * E_tf_t*sp.exp(1j*2*sp.pi/WL*n_low*k_tr_f.dot(l_f*S_tf)) * sp.exp(-1j*t*2*sp.pi))
    E_t_s_time = sp.re(E_inc.norm() * E_ts_t*sp.exp(1j*2*sp.pi/WL*n_high*k_tr_s.dot(l_s*S_ts)) * sp.exp(-1j*t*2*sp.pi) * sp.exp(1j*phase_jump))
    E_inc_s_time = sp.re(E_inc_s0 * sp.exp(-1j*t*2*sp.pi))
    E_inc_p_time = sp.re(E_inc_p0 * sp.exp(-1j*t*2*sp.pi))
    return E_t_f_time + E_t_s_time, E_t_f_time, E_t_s_time, E_inc_s_time, E_inc_p_time

def normalize(x):
    x = sp.Matrix(x)
    x_norm = x.norm()
    return x / x_norm

def normalize_np(x):
    x = np.array(x)
    x_norm = np.linalg.norm(x) + 1e-14
    return x / x_norm

def data_for_cylinder_along_z(center_x,center_y,radius,height_z, WedgeVector):
    """
    

    Parameters
    ----------
    center_x : TYPE
        DESCRIPTION.
    center_y : TYPE
        DESCRIPTION.
    radius : TYPE
        DESCRIPTION.
    height_z : TYPE
        DESCRIPTION.
    WedgeVector : TYPE
        DESCRIPTION.

    Returns
    -------
    x_grid : TYPE
        DESCRIPTION.
    y_grid : TYPE
        DESCRIPTION.
    z_grid : TYPE
        DESCRIPTION.

    """
    z_incl = 2*radius * np.sin(abs(np.pi/2 - np.arccos(np.dot(WedgeVector, normalize_np([WedgeVector[0],WedgeVector[1],0])))))
    z = np.linspace(-z_incl/2, height_z, 200)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    # x = radius*np.cos(theta) + center_x
    # y = radius*np.sin(theta) + center_y
    # r = []
    mask = np.zeros_like(z_grid)
    for i in range(theta_grid.shape[0]):
        for j in range(theta_grid.shape[1]):
            if np.dot( np.array([x_grid[i,j], y_grid[i,j], z_grid[i,j]]) , WedgeVector ) >= 0:
                mask[i,j] = 1
    return x_grid, y_grid, z_grid, mask

def wavefront_coor(h, rot, w, w0, Pos0, x, offset):
    x_R = sp.pi*w0**2*1/w
    Rx = x*(1 + (x_R/x)**2)
    wx = w0*sp.sqrt(1 + (x/x_R)**2)
    # r = {}
    r0 = []
    for i in range(h):
        # r0 = []
        if i == 0:
            if isinstance(x, sp.matrices.dense.MutableDenseMatrix) == True:
                z = sp.zeros(len(x), 1); y = sp.zeros(len(x), 1)
            else:
                z = 0; y = 0
            r0.append([z + offset[0], y + offset[1], x + offset[2]])
        else:
            for j in range(rot):
                if np.any(np.isnan(Rx)) == True:
                    z = (i * wx/h) * sp.cos(j*2*sp.pi/rot)
                    y = (i * wx/h) * sp.sin(j*2*sp.pi/rot)
                else:
                    alpha_max = 0.5 * sp.arcsin(2 * wx/Rx)
                    z = sp.sin(i * alpha_max/h) * Rx * sp.cos(j*2*sp.pi/rot)
                    y = sp.sin(i * alpha_max/h) * Rx * sp.sin(j*2*sp.pi/rot)
                    x = x + Rx - (sp.cos(i * alpha_max/h) * Rx)
                r0.append([z + offset[0], y + offset[1], x + offset[2]])
    #     r[i] = np.array(r0[i])
    return r0

def RotMat(alpha,beta):
    return sp.Matrix([[sp.cos(alpha),-sp.sin(alpha)*sp.sin(beta),sp.sin(alpha)*sp.cos(beta)],
                 [0,sp.cos(beta),sp.sin(beta)],
                 [-sp.sin(alpha),-sp.cos(alpha)*sp.sin(beta),sp.cos(beta)*sp.cos(alpha)]])

def n_e(kin, eps1, eps2, N_vec):
    """
    Calculate refractive indexes of fast and slow ray in biaxial material
    
    Parameters
    ----------
    C : c-axis orientation.
    eps: epsilon-Tensor of material (local).
    AOI : angle of incidence.
    N_vec : normal-vector of surface.

    Returns
    -------
    refractive index (real), refractive index (non-real), refractive index ordinary ray, refraction angle of e-ray

    """
    n_prime = sp.symbols('n_prime')
    k_in = sp.Matrix([kin[0],kin[1],kin[2]])
    N_v = sp.Matrix([N_vec[0],N_vec[1],N_vec[2]])
    
    if np.any(eps1-eps2) == False:
        k = k_in + (-(k_in.T*N_v)[0] - sp.sqrt((k_in.T*N_v)[0]**2 + (n_prime**2-1)))*N_v
    else:
        k = k_in + (-(k_in.T*N_v)[0] + sp.sqrt((k_in.T*N_v)[0]**2 + (n_prime**2-1)))*N_v
        
    k = k/sp.sqrt(sp.nsimplify(k[0]**2 + k[1]**2 + k[2]**2, tolerance=1e-10, rational=True))
    
    K = sp.Matrix([ [0, -k[2], k[1]] , [k[2], 0, -k[0]] , [-k[1], k[0], 0] ])
    
    M = eps2 + (n_prime*K)**2

    sol = sp.solve(M.det(), check = False)
    # sol0 = [sp.N(sol[i]) for i in range(len(sol))]
    # sol = np.real(np.array(sol).astype(np.complex64))
    n_list = [float(sp.N(sp.re(sol[i]))) for i in range(len(sol)) if sp.re(sol[i])>=1.0]
    
    return n_list

def k_t(n_inc,n_list,AOI,kin,N_vec):
    """
    Calculate the k-vector for transmitted beams on surfaces

    Parameters
    ----------
    n0 : Scalar
        refractive index of incoming medium.
    ne2 : List(5)
        refractive indices of exiting medium (ordinary and extraordinary); outcome of n_e(C1, C2, kin, eps1, eps2, AOI, N_vec, typ).
    AOI : Scalar
        angle of incidence.
    kin : Vector(3)
        incoming beam k-vector.
    N_vec : Vector(3)
        surface-vector; pointing away from incoming beam.

    Returns
    -------
    k_to : Vector(3)
        k-vector of ordinary beam.
    k_te : Vector(3)
        k-vector of extraordinary beam.
    theta_o : Scalar
        refraction angle of ordinary beam.

    """
    
    if len(n_list)>1:
        # theta_f = np.arcsin(np.sin(AOI)/n_list[0])
        # theta_s = np.arcsin(np.sin(AOI)/n_list[1])
        k_tf = n_inc*kin + (-n_inc*kin.dot(N_vec) + sp.sqrt(n_inc**2*kin.dot(N_vec)**2 + n_list[0]**2-n_inc**2)) * N_vec
        k_tf = k_tf / k_tf.norm()
        k_ts = n_inc*kin + (-n_inc*kin.dot(N_vec) + sp.sqrt(n_inc**2*kin.dot(N_vec)**2 + n_list[1]**2-n_inc**2)) * N_vec
        k_ts = k_ts / k_ts.norm()
        k = [k_tf, k_ts]
    else:
        k_t = n_inc*kin + (-n_inc*kin.dot(N_vec) + sp.sqrt(n_inc**2*kin.dot(N_vec)**2 + n_list[0]**2-n_inc**2)) * N_vec
        k_t = k_t / k_t.norm()
        k = [k_t]
    return k

def k_r(n_inc,n_list,kin,N_vec):
    """
    Calculate the k-vector for reflected beams on surfaces

    Parameters
    ----------
    ne1 : Scalar
        refractive index of incoming medium.
    ne2 : List(5)
        refractive indices of exiting medium (ordinary and extraordinary); outcome of n_e(C1, C2, kin, eps1, eps2, AOI, N_vec, typ).
    kin : Vector(3)
        incoming beam k-vector.
    N_vec : Vector(3)
        surface-vector; pointing away from incoming beam.

    Returns
    -------
    k_ro : Vector(3)
        k-vector of ordinary beam.
    k_re : Vector(3)
        k-vector of extraordinary beam.

    """

    if len(n_list)>1:
        k_rf = n_inc*kin + (-n_inc*kin.dot(N_vec) - sp.sqrt(n_inc**2*kin.dot(N_vec)**2 + n_list[0]**2-n_inc**2)) * N_vec
        k_rf = k_rf / k_rf.norm()
        k_rs = n_inc*kin + (-n_inc*kin.dot(N_vec) - sp.sqrt(n_inc**2*kin.dot(N_vec)**2 + n_list[1]**2-n_inc**2)) * N_vec
        k_rs = k_rs / k_rs.norm()
        k = [k_rf, k_rs]
    else:
        k_r = n_inc*kin + (-n_inc*kin.dot(N_vec) - sp.sqrt(n_inc**2*kin.dot(N_vec)**2 + n_list[0]**2-n_inc**2)) * N_vec
        k_r = k_r / k_r.norm()
        k = [k_r]
    return k

def Eigen_matrix(kt,eps,n_list):
    """
    Eigenvector-matrix of transmitted electric field

    Parameters
    ----------
    kt : List(3)
        transmitted k-vectors; outcome of k_t(ne1,ne2,AOI,kin,N_vec).
    eps : Matrix(3,3)
        dielectric tensor of transmitting medium in global coordinates.
    ne : List(5)
        refractive indices of exiting medium (ordinary and extraordinary); outcome of n_e(C1, C2, kin, eps1, eps2, AOI, N_vec, typ).

    Returns
    -------
    Matrix(3,3)
        Eigenvector matrix of ordinary electric field.
    Matrix(3,3)
        Eigenvector matrix of extraordinary electric field.
    K_o : Matrix(3,3)
        k-vector matrix of ordinary beam.
    K_e : Matrix(3,3)
        k-vector matrix of extraordinary beam.

    """
    if len(n_list)>1:
        K_f = sp.Matrix([[0,-kt[0][2],kt[0][1]], [kt[0][2],0,-kt[0][0]], [-kt[0][1],kt[0][0],0]])
        K_s = sp.Matrix([[0,-kt[1][2],kt[1][1]], [kt[1][2],0,-kt[1][0]], [-kt[1][1],kt[1][0],0]])
        K = eps + (n_list[0]*K_f)**2, eps + (n_list[1]*K_s)**2 ,K_f, K_s
    else:
        K_t = sp.Matrix([[0,-kt[0][2],kt[0][1]], [kt[0][2],0,-kt[0][0]], [-kt[0][1],kt[0][0],0]])
        K = eps + (n_list[0]*K_t)**2, K_t
    return K

def Eigen_matrix_ref(kref,eps,n_list):
    """
    Eigenvector-matrix of reflected electric field

    Parameters
    ----------
    kref : List(2)
        reflected k-vectors; outcome of k_r(ne1,ne2,kin,N_vec).
    eps : Matrix(3,3)
        dielectric tensor of reflection medium in global coordinates.
    ne : List(5)
        refractive indices of exiting medium (ordinary and extraordinary); outcome of n_e(C1, C2, kin, eps1, eps2, AOI, N_vec, typ).

    Returns
    -------
    Matrix(3,3)
        Eigenvector matrix of ordinary electric field.
    Matrix(3,3)
        Eigenvector matrix of extraordinary electric field.
    K_o : Matrix(3,3)
        k-vector matrix of ordinary beam.
    K_e : Matrix(3,3)
        k-vector matrix of extraordinary beam.

    """
    if len(n_list)>1:
        K_f = sp.Matrix([[0,-kref[0][2],kref[0][1]], [kref[0][2],0,-kref[0][0]], [-kref[0][1],kref[0][0],0]])
        K_s = sp.Matrix([[0,-kref[1][2],kref[1][1]], [kref[1][2],0,-kref[1][0]], [-kref[1][1],kref[1][0],0]])
        K = eps + (n_list[0]*K_f)**2, eps + (n_list[1]*K_s)**2, K_f, K_s
    else:
        K_r = sp.Matrix([[0,-kref[0][2],kref[0][1]], [kref[0][2],0,-kref[0][0]], [-kref[0][1],kref[0][0],0]])
        K = eps + (n_list[0]*K_r)**2, K_r
    return K

def C_vector(N_vec,s1,s2,K_inc,E_inc_s,E_inc_p,H_inc_s,H_inc_p):
    """
    Support-vector for calculating the transmission and reflection coefficients for ordinary and extraordinary beams. Contains tangential and normal components of incoming electric and magnetic fields

    Parameters
    ----------
    N_vec : Vector(3)
        surface-vector; pointing away from incoming beam.
    s1 : Vector(3)
        tangential component extraction.
    s2 : Vector(3)
        normal component extraction.
    ne : List(5)
        refractive indices of exiting medium (ordinary and extraordinary); outcome of n_e(C1, C2, kin, eps1, eps2, AOI, N_vec, typ).
    K_inc : Matrix(3,3)
        k-vector matrix of incoming beam.
    s_vec : Vector(3)
        s-orientation of incoming field.
    p_vec : Vector(3)
        p-orientation of incoming field.

    Returns
    -------
    C_s : Vector(4)
        C-vector of ordinary beam.
    C_p : Vector(4)
        C-vector of extraordinary beam.

    """
    E_inc = E_inc_s + E_inc_p
    H_inc = H_inc_s + H_inc_p
    # H_inc_p = np.dot(n_list[0]*K_inc, fast)#/np.linalg.norm(np.dot(ne[0]*K_inc, E_inc_p))
    # H_inc_s = np.dot(n_list[1]*K_inc, slow)#/np.linalg.norm(np.dot(ne[2]*K_inc, E_inc_s))
    C_s = sp.Matrix([s1.dot(E_inc_s), s2.dot(E_inc_s), s1.dot(H_inc_s), s2.dot(H_inc_s)])
    C_p = sp.Matrix([s1.dot(E_inc_p), s2.dot(E_inc_p), s1.dot(H_inc_p), s2.dot(H_inc_p)])
    C = sp.Matrix([s1.dot(E_inc), s2.dot(E_inc), s1.dot(H_inc), s2.dot(H_inc)])
    return C_s, C_p, C

#%%
# =============================================================================
# Dictonaries
# =============================================================================

kloc = {}
kin = {}
theta_inc = {}
k_Rot = {}
s_vec = {}
n_tr = {}
S_inc = {}
S_tf = {}
S_ts = {}
S_ts_f = {}
S_ts_s = {}
k_tf = {}
k_ts = {}
E_inc = {}
E_inc_s0 = {}
E_inc_p0 = {}
E_tf_t = {}
E_ts_t = {}
E_tf = {}
E_ts = {}
E_end = {}
PlanePoints = {}

#%%
# =============================================================================
# Orientation of the crystal
# =============================================================================

local_N_vec = sp.Matrix([0,0,1])               # Normal Vector of surface in global coordinates
center = ['0', '0', '0.075']                   # center-point of the crystal (dimensions in meter)
L = 2 * sp.Float(center[2], p)                    # length of the crystal
nSa = '1.754'                                  # refractive index of Fast-axis of biaxial medium
nMa = '1.754'                                  # refractive index of Medium-axis of biaxial medium
nFa = '1.747'                                  # refractive index of Slow-axis of biaxial medium
Dn = '2.8e-6'

#%%
# =============================================================================
# Defining incoming beam and orientation
# =============================================================================
"""angle of k in xz plane (0 --> z) """
alpha = mpmath.radians(sp.Float('0.0', p))
"""angle of k from xz plane to y (0 --> xz)"""                                                                
beta =  mpmath.radians(sp.Float('0.0', p))
""""angle of Ca in medium 1 in xz plane (0 --> z)"""
gamma1 = mpmath.radians(sp.Float('0.0', p))
""""angle of Ca in medium 1 from xz plane to y (0 --> xz)"""
delta1 = mpmath.radians(sp.Float('0.0', p))
""""angle of Ca in medium 2 in xz plane (0 --> z) """
gamma2 = mpmath.radians(sp.Float('0.0', p))
""""angle of Ca in medium 2 from xz plane to y (0 --> xz)"""
delta2 = mpmath.radians(sp.Float('0.0', p))
""""angle of N_vec in xz plane"""
epsilon = mpmath.radians(sp.Float('0.0', p))
""""angle of N_vec from xz plane to y"""
zeta = mpmath.radians(sp.Float('0.0', p))

WL = sp.Float('1.064e-6', p)
w = 2*sp.pi*299792458/WL
InputBeamPoints = wavefront_coor(1, 1, WL, 0.00005, 0, -0.5, [0, 0, 0.4])
dInputBeamPoints = wavefront_coor(1, 1, WL, 0.00005, 0, 0.01, [0, 0, 0.4])

for i in range(len(InputBeamPoints)):
    kloc[i+1] = normalize(sp.Matrix(dInputBeamPoints[i]) + -1*sp.Matrix(InputBeamPoints[i]) )
# kloc[1] = normalize([0,0,1])
# kloc[2] = normalize([-0.05,0,1])

RotMk = RotMat(alpha, beta)
RotMN1 = RotMat(gamma1-epsilon,delta1-zeta)
RotMN2 = RotMat(gamma2-epsilon,delta2-zeta)
RotMNv = RotMat(epsilon,zeta)
RotMNz = R.as_matrix(R.from_euler('z', 20, degrees = True))

eps2 = sp.Matrix([[sp.Float(nSa, p)**2,0,0], [0,(sp.Float(nMa, p) - sp.Float(Dn, p))**2,0], [0,0,sp.Float(nFa, p)**2]])
eps_sap = RotMNv * (RotMN2 * (eps2 * RotMN2.T) * RotMNv.T)
eps2 = RotMNz * (eps_sap * RotMNz.T)

eps1 = sp.Matrix([[1**2,0,0], [0,1**2,0], [0,0,1**2]])
eps_air = RotMNv * ((RotMN1 * (eps1 * RotMN1)) * RotMNv.T)

for i in kloc:

    N_vec_inc = RotMNv * local_N_vec
    N_vec_out = local_N_vec.copy()
    kin[i] = RotMk * kloc[i]

    if (N_vec_inc.cross(kin[i])).norm() == 0:
        s_vec[i] = RotMk * sp.Matrix([1,0,0])
    else:
        s_vec[i] = normalize(kin[i].cross(N_vec_inc))
    p_vec = s_vec[i].cross(kin[i])

for i in kloc:
    if len(kloc) > 1:
        theta_inc[i] = 0#-np.arccos(np.dot(s_vec[1], s_vec[i]))
        k_Rot_beta = sp.acos((sp.Matrix([0,0,1]).T * normalize([0 , kin[i][1] , kin[i][2]]))[0])
        k_Rot_alpha = sp.acos((normalize([0, kin[i][1] , kin[i][2]]).T * kin[i])[0])
        if kin[i][0] < 0:
            k_Rot_alpha = -k_Rot_alpha
        if kin[i][1] < 0:
            k_Rot_beta = -k_Rot_beta
        k_Rot[i] = [k_Rot_alpha, k_Rot_beta]
    else:
        theta_inc[i] = mpmath.radians(sp.Float('0', p))
        k_Rot[i] = [0, 0]
    K_inc = sp.Matrix([[0,-kin[i][2],kin[i][1]], [kin[i][2],0,-kin[i][0]], [-kin[i][1],kin[i][0],0]])

#%%
# =============================================================================
# Code
# =============================================================================

    """ Propagation from isotropic-medium 1 to biaxial medium 2 """

    n_inc = 1.0
    I_inc = 1.0
    
    n_tr[i] = n_e(kin[i], eps_air, eps2, N_vec_inc)
    # if np.linalg.norm(np.cross(kin, N_vec)) > 0:
    #     n_tr = list(dict.fromkeys(n_tr))
    n_re = n_e(kin[i], eps_air, eps_air, N_vec_inc)
    n_re = list(dict.fromkeys(n_re))
    
    # defining field orientation in global coordinates
    E_inc_p0[i] = sp.sin(theta_inc[i]) * p_vec
    E_inc_s0[i] = sp.cos(theta_inc[i]) * s_vec[i]
    E_inc[i] = E_inc_p0[i] + E_inc_s0[i]
    H_inc = (n_inc*K_inc) * E_inc[i]
    H_inc_s = (n_inc*K_inc) * E_inc_s0[i]
    H_inc_p = (n_inc*K_inc) * E_inc_p0[i]
    S_inc[i] = sp.re(E_inc[i].cross(H_inc.conjugate())) / sp.re(E_inc[i].cross(H_inc.conjugate()).norm())
    
    AOI = sp.acos((-N_vec_inc.T * S_inc[i])[0])
    k_tr = k_t(n_inc,n_tr[i],AOI,kin[i],N_vec_inc)
    k_re = k_r(n_inc,n_re,kin[i],N_vec_inc)
    
    EMT = Eigen_matrix(k_tr,eps2,n_tr[i])
    EMR = Eigen_matrix_ref(k_re,eps1,n_re)
    
    if np.all(EMT[0]==EMT[1]) == True:
        SVD_tf = mpmath.svd(EMT[0])
        SVD_ts = mpmath.svd(EMT[1])
        E_tf[i] = SVD_tf[0][:,2]
        E_ts[i] = SVD_ts[0][:,1]
    else:
        SVD_tf = mpmath.svd(EMT[0])
        SVD_ts = mpmath.svd(EMT[1])
        E_tf[i] = SVD_tf[0][:,2]
        E_ts[i] = SVD_ts[0][:,2]
    
    SVD_r = mpmath.svd(EMR[0])
    E_rs = SVD_r[0][:,2]
    E_rp = SVD_r[0][:,1]
    if sp.Abs(E_rs.dot(s_vec[i])) < 1e-14:
        E_rs = s_vec[i].copy()
        E_rp = SVD_r[0][:,2]
    
    E_tf0 = E_tf[i].copy()
    E_ts0 = E_ts[i].copy()
    
    H_tf = (n_tr[i][0]*EMT[2]) * E_tf[i]
    H_ts = (n_tr[i][1]*EMT[3]) * E_ts[i]
    H_rs = (n_re[0]*EMR[1]) * E_rs
    H_rp = (n_re[0]*EMR[1]) * E_rp
    
    S_tf[i] = normalize( sp.re( E_tf[i].cross(H_tf.C) ) )
    S_ts[i] = normalize( sp.re( E_ts[i].cross(H_ts.C) ) )
    S_rs = normalize( sp.re( E_rs.cross(H_rs.C) ) )
    S_rp = normalize( sp.re( E_rp.cross(H_rp.C) ) )
    S = [S_inc[i],S_tf[i], S_ts[i], S_rs, S_rp]
    
    if (kin[i].cross( N_vec_inc )).norm() == 0:
        s1 = RotMk * sp.Matrix([1,0,0])
    else:
        s1 = kin[i].cross( N_vec_inc )
    s2 = N_vec_inc.cross( s1 )
    
    F = sp.Matrix([[s1.dot(E_tf[i]), s1.dot(E_ts[i]), -s1.dot(E_rs), -s1.dot(E_rp)],
                  [s2.dot(E_tf[i]), s2.dot(E_ts[i]), -s2.dot(E_rs), -s2.dot(E_rp)],
                  [s1.dot(H_tf), s1.dot(H_ts), -s1.dot(H_rs), -s1.dot(H_rp)],
                  [s2.dot(H_tf), s2.dot(H_ts), -s2.dot(H_rs), -s2.dot(H_rp)]])
    
    if (S_inc[i].cross( N_vec_inc )).norm() == 0:
        E_inc_s_norm = RotMk * sp.Matrix([1,0,0])
    else:
        E_inc_s_norm = normalize(S_inc[i].cross( N_vec_inc ))
    E_inc_p_norm = S_inc[i].cross( E_inc_s_norm )
    H_inc_s_norm = (n_inc*K_inc) * E_inc_s_norm
    H_inc_p_norm = (n_inc*K_inc) * E_inc_p_norm
    
    A_s = F.inv() * C_vector(N_vec_inc,s1,s2,K_inc,E_inc_s_norm,E_inc_p_norm,H_inc_s_norm,H_inc_p_norm)[0]
    A_f = F.inv() * C_vector(N_vec_inc,s1,s2,K_inc,E_inc_s_norm,E_inc_p_norm,H_inc_s_norm,H_inc_p_norm)[1]
    
    a_stf = A_s[0]            # transmission coefficient to fast-ray
    a_sts = A_s[1]            # transmission coefficient to slow-ray
    a_srs = A_s[2]            # reflection coefficient to s-pol
    a_srp = A_s[3]            # reflection coefficient to p-pol
    
    a_ptf = A_f[0]            # transmission coefficient to fast-ray
    a_pts = A_f[1]            # transmission coefficient to slow-ray
    a_prs = A_f[2]            # reflection coefficient to s-pol
    a_prp = A_f[3]            # reflection coefficient to p-pol
    
    P_tf = sp.Matrix([a_stf*E_tf[i].T, a_ptf*E_tf[i].T, S_tf[i].T]).T * sp.Matrix([E_inc_s_norm.T, E_inc_p_norm.T, S_inc[i].T])
    P_ts = sp.Matrix([a_sts*E_ts[i].T, a_pts*E_ts[i].T, S_ts[i].T]).T * sp.Matrix([E_inc_s_norm.T, E_inc_p_norm.T, S_inc[i].T])
    # P_rs = np.dot(np.transpose(np.array([a_srs*E_rs, a_prs*E_rs, S_rs])), (np.array([E_inc_s_norm, E_inc_p_norm, S_inc])))
    # P_rp = np.dot(np.transpose(np.array([a_srp*E_rs, a_prp*E_rp, S_rp])), (np.array([E_inc_s_norm, E_inc_p_norm, S_inc])))
    P_r = sp.Matrix([a_srs*E_rs.T, a_prp*E_rp.T, S_rs.T]).T * sp.Matrix([E_inc_s_norm.T, E_inc_p_norm.T, S_inc[i].T])
    
    E_t_f = P_tf * E_inc[i]
    E_t_s = P_ts * E_inc[i]
    # E_r_s = np.dot(P_rs, E_inc[i])
    # E_r_p = np.dot(P_rp, E_inc[i])
    E_r = P_r * E_inc[i]
    
    I_t_f = n_tr[i][0] * S_tf[i].dot(N_vec_inc) / n_inc / S_inc[i].dot(N_vec_inc) * E_t_f.norm()**2 * I_inc / E_inc[i].norm()**2
    I_t_s = n_tr[i][1] * S_ts[i].dot(N_vec_inc) / n_inc / S_inc[i].dot(N_vec_inc) * E_t_s.norm()**2 * I_inc / E_inc[i].norm()**2
    
    # I_r_s = np.dot(S_rs,N_vec)/np.dot(S_inc,N_vec) * np.linalg.norm(E_r_s)**2 * I_inc / np.linalg.norm(E_inc[i])**2
    # I_r_p = np.dot(S_rp,N_vec)/np.dot(S_inc,N_vec) * np.linalg.norm(E_r_p)**2 * I_inc / np.linalg.norm(E_inc[i])**2
    I_r = S_rs.dot(N_vec_inc) / S_inc[i].dot(N_vec_inc) * E_r.norm()**2 * I_inc / E_inc[i].norm()**2
    I0 = [I_inc, I_t_f, I_t_s, I_r]
    
    df0 = pd.DataFrame([str(np.array(E_inc[i].T).astype(np.float64)), str(n_tr[i]), str(n_re), str(np.array(E_tf[i].T).astype(np.float64)), str(E_ts[i]), str(E_rs), str(E_rp), str(S[0]), str(S[1]), str(S[2]), str(S[3]), str(S[4]), str(E_t_f), str(E_t_s), str(E_r)] , ['incoming E-field','refractive index transmission', 'refractive index reflection', 'transmitted E-field direction (fast)', 'transmitted E-field direction (slow)', 'reflected E-field direction (s)', 'reflected E-field direction (p)','Poynting vector (incoming)','Poynting vector (fast)','Poynting vector (slow)','Poynting vector (s-reflected)','Poynting vector (p-reflected)','transmitted E-field (fast)', 'transmitted E-field (slow)','reflected E-field'], columns = [''])
    print('')
    print('Isotropic Medium 1 to Biaxial Medium 2:')
    print()
    print('incoming E-field:')
    sp.printing.pprint(E_inc[i])
    print()
    print('refractive index transmission:')
    sp.printing.pprint(n_tr[i])
    print()
    print('refractive index reflection:')
    sp.printing.pprint(n_re)
    print()
    print('transmitted E-field direction (fast):')
    sp.printing.pprint(E_tf[i])
    print()
    print('transmitted E-field direction (slow):')
    sp.printing.pprint(E_ts[i])
    print()
    print('reflected E-field direction (s):')
    sp.printing.pprint(E_rs)
    print()
    print('reflected E-field direction (p):')
    sp.printing.pprint(E_rp)
    print()
    print('Poynting vector (incoming):')
    sp.printing.pprint(S[0])
    print()
    print('Poynting vector (fast):')
    sp.printing.pprint(S[1])
    print()
    print('Poynting vector (slow):')
    sp.printing.pprint(S[2])
    print()
    print('Poynting vector (s-reflected):')
    sp.printing.pprint(S[3])
    print()
    print('Poynting vector (p-reflected):')
    sp.printing.pprint(S[4])
    print()
    print('transmitted E-field (fast):')
    sp.printing.pprint(E_t_f)
    print()
    print('transmitted E-field (slow):')
    sp.printing.pprint(E_t_s)
    print()
    print('reflected E-field:')
    sp.printing.pprint(E_r)
    print()
    
    """ Propagation of fast ray from biaxial medium 2 to isotropic medium 3 """
    
    n_inc = n_tr[i][0]
    I_inc = I_t_f
    
    n_tr_f = n_e(k_tr[0], eps2, eps1, N_vec_out)
    n_tr_f = list(dict.fromkeys(n_tr_f))
    n_re_f = n_e(k_tr[0], eps2, eps2, N_vec_out)
    # if np.linalg.norm(np.cross(kin, N_vec)) > 0:
    #     n_re_f = list(dict.fromkeys(n_re_f))
    
    AOI_f = sp.acos(-N_vec_out.dot(S_tf[i]))
    k_tr_f = k_t(n_inc,n_tr_f,AOI_f,k_tr[0],N_vec_out)
    k_re_f = k_r(n_inc,n_re_f,k_tr[0],N_vec_out)
    
    EMT_f = Eigen_matrix(k_tr_f,eps1,n_tr_f)
    EMR_f = Eigen_matrix_ref(k_re_f,eps2,n_re_f)
    
    E_inc_f = E_t_f
    E_inc_s1 = normalize( S_tf[i].cross(E_t_f) )                   # Pseudo slow-field with no power
    H_inc_f = n_inc*EMT_f[1] * E_inc_f
    H_inc_s = n_inc*EMT_f[1] * E_inc_s1
    
    SVD_t = mpmath.svd(EMT_f[0])
    if len(EMR_f) > 2:
        SVD_rs = mpmath.svd(EMR_f[0])
        SVD_rp = mpmath.svd(EMR_f[1])
    else:
        SVD_rs = mpmath.svd(EMR_f[0])
        SVD_rp = mpmath.svd(EMR_f[0])
    E_f_ts = SVD_t[0][:,2]
    E_f_tp = SVD_t[0][:,1]
    if sp.Abs(E_f_ts.dot(s_vec[i])) < 1e-14:
        E_f_ts = SVD_t[0][:,1]
        E_f_tp = SVD_t[0][:,2]
    
    if SVD_rs[0][:,2].dot(E_inc_f) < 1:
        E_rf0 = E_tf0.copy()
    else:
        E_rf0 = SVD_rs[0][:,2]
    if SVD_rp[0][:,2].dot(E_inc_s1) < 1:
        E_rs0 = E_ts0.copy()
    else:
        E_rs0 = SVD_rp[0][:,2]
    
    H_ts = n_tr_f[0]*EMT_f[1] * E_f_ts
    H_tp = n_tr_f[0]*EMT_f[1] * E_f_tp
    if len(EMR_f) > 2:
        H_rf = n_re_f[0]*EMR_f[2] * E_rf0
        H_rs = n_re_f[1]*EMR_f[3] * E_rs0
    else:
        H_rf = n_re_f[0]*EMR_f[1] * E_rf0
        H_rs = n_re_f[0]*EMR_f[1] * E_rs0
    
    S_ts_f[i] = normalize( sp.re( E_f_ts.cross(H_ts.C) ) )
    S_tp_f = normalize( sp.re( E_f_tp.cross(H_tp.C) ) )
    S_rf_f = normalize( sp.re( E_rf0.cross(H_rf.C) ) )
    S_rs_f = normalize( sp.re( E_rs0.cross(H_rs.C) ) )
    S = [S_tf[i],S_ts_f[i], S_tp_f, S_rf_f, S_rs_f]
    
    if (k_tr[0].cross(N_vec_out)).norm() == 0:
        s1 = RotMk * sp.Matrix([1,0,0])
    else:
        s1 = k_tr[0].cross(N_vec_out)
    s2 = N_vec_out.cross(s1)
    
    F = sp.Matrix([[s1.dot(E_f_ts), s1.dot(E_f_tp), -s1.dot(E_rf0), -s1.dot(E_rs0)],
                  [s2.dot(E_f_ts), s2.dot(E_f_tp), -s2.dot(E_rf0), -s2.dot(E_rs0)],
                  [s1.dot(H_ts), s1.dot(H_tp), -s1.dot(H_rf), -s1.dot(H_rs)],
                  [s2.dot(H_ts), s2.dot(H_tp), -s2.dot(H_rf), -s2.dot(H_rs)]])
    
    E_inc_f_norm = normalize( E_inc_f )
    E_inc_s_norm = normalize( E_inc_s1 )
    # H_inc_f_norm = H_inc_f/(np.linalg.norm(H_inc_f) + 1e-14)
    # H_inc_s_norm = H_inc_s/(np.linalg.norm(H_inc_s) + 1e-14)
    H_inc_f_norm = n_inc*EMT_f[1] * E_inc_f_norm
    H_inc_s_norm = n_inc*EMT_f[1] * E_inc_s_norm
    
    
    # s_vec_f = np.cross(S_tf[i],N_vec) / np.linalg.norm(np.cross(S_tf[i],N_vec))
    # alpha = np.arccos(np.dot(E_inc_f/np.linalg.norm(E_inc_f + 1e-14),s_vec_f))
    # E_inc_f_s = np.cos(alpha)*s_vec_f * np.linalg.norm(E_inc_f)
    # E_inc_f_p = np.sin(alpha)*np.cross(s_vec_f, S_tf[i]) * np.linalg.norm(E_inc_f)
    
    # Fr0 = Fresnel(n_inc, n_tr_f[0], S_tf[i], S_ts_f[i], N_vec)
    # if np.dot(E_inc_f,s_vec) < 1e-14:
    #     A_f_f = [0,Fr0[1],Fr0[3],0]
    # else:
    #     A_f_f = [Fr0[0],0,0,Fr0[2]]
    
    A_f_f = F.inv() * C_vector(N_vec_out,s1,s2,EMT_f[1],E_inc_f_norm,E_inc_s_norm,H_inc_f_norm,H_inc_s_norm)[0]
    A_s_f = sp.Matrix([0,0,0,0])
    
    a_fts = A_f_f[0]            # transmission coefficient to s-pol
    a_ftp = A_f_f[1]            # transmission coefficient to p-pol
    a_frf = A_f_f[2]            # reflection coefficient to fast-ray
    a_frs = A_f_f[3]            # reflection coefficient to slow-ray
    
    a_sts = A_s_f[0]            # transmission coefficient to s-pol
    a_stp = A_s_f[1]            # transmission coefficient to p-pol
    a_srf = A_s_f[2]            # reflection coefficient to fast-ray
    a_srs = A_s_f[3]            # reflection coefficient to slow-ray
    
    P_t = (sp.Matrix([(a_fts*E_f_ts + a_ftp*E_f_tp).T, sp.zeros(3,1).T, S_ts_f[i].T])).T * sp.Matrix([E_inc_f_norm.T, E_inc_s_norm.T, S_tf[i].T])
    P_rf = (sp.Matrix([a_frf*E_rf0.T ,sp.zeros(3,1).T, S_rf_f.T])).T * sp.Matrix([E_inc_f_norm.T, E_inc_s_norm.T, S_tf[i].T])
    P_rs = (sp.Matrix([a_frs*E_rs0.T ,sp.zeros(3,1).T, S_rs_f.T])).T * sp.Matrix([E_inc_f_norm.T, E_inc_s_norm.T, S_tf[i].T])
    
    E_tf_t[i] = P_t * E_inc_f
    E_rf_f = P_rf * E_inc_f
    E_rf_s = P_rs * E_inc_f
    
    # E_tf_t[i] = np.add(Fr0[0]*np.linalg.norm(E_inc_f_s)*s_vec , Fr0[1]*np.linalg.norm(E_inc_f_p)*p_vec)
    
    I_t = n_tr_f[0]*S_ts_f[i].dot(N_vec_out) / n_inc / S_tf[i].dot(N_vec_out) * E_tf_t[i].norm()**2 * I_inc / E_inc_f.norm()**2
    I_r_f = S_rf_f.dot(N_vec_out) / S_tf[i].dot(N_vec_out) * E_rf_f.norm()**2 * I_inc / E_inc_f.norm()**2
    I_r_s = S_rs_f.dot(N_vec_out) / S_tf[i].dot(N_vec_out) * E_rf_s.norm()**2 * I_inc / E_inc_f.norm()**2
    I1 = [I_inc, I_t, I_r_f, I_r_s]
    
    df1 = pd.DataFrame([str(E_inc_f), str(n_tr_f), str(n_re_f), str(E_f_ts), str(E_f_tp), str(E_rf0), str(E_rs0), str(S[0]), str(S[1]), str(S[2]), str(S[3]), str(S[4]), str(E_tf_t[i]),  str(E_rf_f), str(E_rf_s)] , ['E-field (fast)','refractive index transmission', 'refractive index reflection', 'transmitted E-field direction (s)', 'transmitted E-field direction (p)', 'reflected E-field direction (fast)', 'reflected E-field direction (slow)','Poynting vector (fast)','Poynting vector (s)','Poynting vector (p)','Poynting vector (fast-reflected)','Poynting vector (slow-reflected)','transmitted E-field','reflected E-field (fast)','reflected E-field (slow)'], columns = [''])
    print('')
    print('Biaxial Medium 2 (fast) to Isotropic Medium 3:')
    print(df1)
    
    """ Propagation of slow ray from biaxial medium 2 to isotropic medium 3 """
    
    n_inc = n_tr[i][1]
    I_inc = I_t_s
    
    n_tr_s = n_e(k_tr[1], eps2, eps1, N_vec_out)
    n_tr_s = list(dict.fromkeys(n_tr_s))
    n_re_s = n_e(k_tr[1], eps2, eps2, N_vec_out)
    # if np.linalg.norm(np.cross(kin, N_vec)) > 0:
    #     n_re_s = list(dict.fromkeys(n_re_s))
    
    AOI_s = sp.acos(-N_vec_out.dot(S_ts[i]))
    k_tr_s = k_t(n_inc,n_tr_s,AOI_s,k_tr[1],N_vec_out)
    k_re_s = k_r(n_inc,n_re_s,k_tr[1],N_vec_out)
    
    EMT_s = Eigen_matrix(k_tr_s,eps1,n_tr_s)
    EMR_s = Eigen_matrix_ref(k_re_s,eps2,n_re_s)
    
    E_inc_s = E_t_s.copy()
    E_inc_f1 = normalize( S_ts[i].cross(E_t_s) )                 # Pseudo fast-field with no power
    H_inc_s = n_inc*EMT_f[1] * E_inc_s
    H_inc_f = n_inc*EMT_f[1] * E_inc_f1
    
    SVD_t = mpmath.svd(EMT_s[0])
    if len(EMR_s) > 2:
        SVD_rs = mpmath.svd(EMR_s[0])
        SVD_rp = mpmath.svd(EMR_s[1])
    else:
        SVD_rs = mpmath.svd(EMR_s[0])
        SVD_rp = mpmath.svd(EMR_s[0])
    E_s_ts = SVD_t[0][:,2]
    E_s_tp = SVD_t[0][:,1]
    if sp.Abs(E_s_ts.dot(s_vec[i])) < 1e-14:
        E_s_ts = SVD_t[0][:,1]
        E_s_tp = SVD_t[0][:,2]
    
    if sp.Abs(SVD_rs[0][:,2].dot(E_inc_f1)) < 1e-14:
        E_rf1 = E_tf0.copy()
    else:
        E_rf1 = SVD_rs[0][:,2]
    if sp.Abs(SVD_rp[0][:,2].dot(E_inc_s)) < 1e-14:
        E_rs1 = E_ts0.copy()
    else:
        E_rs1 = SVD_rp[0][:,2]
    
    H_ts = n_tr_s[0]*EMT_s[1] * E_s_ts
    H_tp = n_tr_s[0]*EMT_s[1] * E_s_tp
    if len(EMR_s) > 2:
        H_rf = n_re_s[0]*EMR_s[2] * E_rf1
        H_rs = n_re_s[1]*EMR_s[3] * E_rs1
    else:
        H_rf = n_re_s[0]*EMR_s[1] * E_rf1
        H_rs = n_re_s[0]*EMR_s[1] * E_rs1
    
    S_ts_s[i] = normalize( sp.re( E_s_ts.cross(H_ts.C) ) )
    S_tp_s = normalize( sp.re( E_s_tp.cross(H_tp.C) ) )
    S_rf_s = normalize( sp.re( E_rf1.cross(H_rf.C) ) )
    S_rs_s = normalize( sp.re( E_rs1.cross(H_rs.C) ) )
    S = [S_ts[i],S_ts_s[i], S_tp_s, S_rf_s, S_rs_s]
    
    if (k_tr[1].cross(N_vec_out)).norm() == 0:
        s1 = RotMk * sp.Matrix([1,0,0])
    else:
        s1 = k_tr[1].cross(N_vec_out)
    s2 = N_vec_out.cross(s1)
    
    F = sp.Matrix([[s1.dot(E_s_ts), s1.dot(E_s_tp), -s1.dot(E_rf1), -s1.dot(E_rs1)],
                  [s2.dot(E_s_ts), s2.dot(E_s_tp), -s2.dot(E_rf1), -s2.dot(E_rs1)],
                  [s1.dot(H_ts), s1.dot(H_tp), -s1.dot(H_rf), -s1.dot(H_rs)],
                  [s2.dot(H_ts), s2.dot(H_tp), -s2.dot(H_rf), -s2.dot(H_rs)]])
    
    E_inc_s_norm = normalize( E_inc_s )
    E_inc_f_norm = normalize( E_inc_f1 )
    # H_inc_f_norm = H_inc_f/(np.linalg.norm(H_inc_f) + 1e-14)
    # H_inc_s_norm = H_inc_s/(np.linalg.norm(H_inc_s) + 1e-14)
    H_inc_f_norm = n_inc*EMT_s[1] * E_inc_f_norm
    H_inc_s_norm = n_inc*EMT_s[1] * E_inc_s_norm    
    
    # s_vec_s = np.cross(S_ts[i],N_vec) / np.linalg.norm(np.cross(S_ts[i],N_vec))
    # alpha = np.arccos(np.dot(E_inc_f/np.linalg.norm(E_inc_s + 1e-14),s_vec_s))
    # E_inc_s_s = np.cos(alpha)*s_vec_s * np.linalg.norm(E_inc_s)
    # E_inc_s_p = np.sin(alpha)*np.cross(s_vec_s, S_ts[i]) * np.linalg.norm(E_inc_s)
    
    # Fr1 = Fresnel(n_inc, n_tr_s[0], S_ts[i], S_ts_s[i], N_vec)
    # if np.dot(E_inc_s,s_vec) < 1e-14:
    #     A_s_s = [0,Fr1[1],Fr1[3],0]
    # else:
    #     A_s_s = [Fr1[0],0,0,Fr1[2]]
      
    A_s_s = F.inv() * C_vector(N_vec_out,s1,s2,EMT_s[1],E_inc_f_norm,E_inc_s_norm,H_inc_f_norm,H_inc_s_norm)[1]
    A_s_f = sp.Matrix([0,0,0,0])
    
    a_fts = A_s_f[0]            # transmission coefficient to s-pol
    a_ftp = A_s_f[1]            # transmission coefficient to p-pol
    a_frf = A_s_f[2]            # reflection coefficient to fast-ray
    a_frs = A_s_f[3]            # reflection coefficient to slow-ray
    
    a_sts = A_s_s[0]            # transmission coefficient to s-pol
    a_stp = A_s_s[1]            # transmission coefficient to p-pol
    a_srf = A_s_s[2]            # reflection coefficient to fast-ray
    a_srs = A_s_s[3]            # reflection coefficient to slow-ray
    
    P_t = (sp.Matrix([(a_sts*E_s_ts + a_stp*E_s_tp).T, sp.zeros(3,1).T, S_ts_s[i].T])).T * sp.Matrix([E_inc_s_norm.T, E_inc_f_norm.T, S_ts[i].T])
    P_rf = (sp.Matrix([a_srf*E_rf1.T, sp.zeros(3,1).T, S_rf_s.T])).T * sp.Matrix([E_inc_s_norm.T, E_inc_f_norm.T, S_ts[i].T])
    P_rs = (sp.Matrix([a_srs*E_rs1.T, sp.zeros(3,1).T, S_rs_s.T])).T * sp.Matrix([E_inc_s_norm.T, E_inc_f_norm.T, S_ts[i].T])
    
    E_ts_t[i] = P_t * E_inc_s
    E_rs_f = P_rf * E_inc_s
    E_rs_s = P_rs * E_inc_s
    
    # E_ts_s = np.add(Fr1[0]*np.linalg.norm(E_inc_s_s)*s_vec , Fr1[1]*np.linalg.norm(E_inc_s_p)*p_vec)
    
    I_t = n_tr_s[0]*S_ts_s[i].dot(N_vec_out) / n_inc / S_ts[i].dot(N_vec_out) * E_ts_t[i].norm()**2 * I_inc / E_inc_s.norm()**2
    I_r_f = S_rf_s.dot(N_vec_out) / S_ts[i].dot(N_vec_out) * E_rs_f.norm()**2 * I_inc / E_inc_s.norm()**2
    I_r_s = S_rs_s.dot(N_vec_out) / S_ts[i].dot(N_vec_out) * E_rs_s.norm()**2 * I_inc / E_inc_s.norm()**2
    I2 = [I_inc, I_t, I_r_f, I_r_s]
    
    df2 = pd.DataFrame([str(E_inc_s), str(n_tr_s), str(n_re_s), str(E_s_ts), str(E_s_tp), str(E_rf1), str(E_rs1), str(S[0]), str(S[1]), str(S[2]), str(S[3]), str(S[4]), str(E_ts_t[i]),  str(E_rs_f), str(E_rs_s)] , ['E-field (slow)','refractive index transmission', 'refractive index reflection', 'transmitted E-field direction (s)', 'transmitted E-field direction (p)', 'reflected E-field direction (fast)', 'reflected E-field direction (slow)','Poynting vector (fast)','Poynting vector (s)','Poynting vector (p)','Poynting vector (fast-reflected)','Poynting vector (slow-reflected)','transmitted E-field','reflected E-field (fast)','reflected E-field (slow)'], columns = [''])
    print('')
    print('Biaxial Medium 2 (slow) to Isotropic Medium 3:')
    print(df2)
    
    df3 = pd.DataFrame([I0[0], I1[1]+I2[1], I0[3]+I1[2]+I2[2]+I1[3]+I2[3]],['incoming Intensity', 'transmitted Intensity', 'reflected Intensity'],columns=[''])
    print('')
    print('Intensities:')
    print(df3)

#%%
# =============================================================================
# Assignment
# =============================================================================
    
    #n_tr[i] = (np.array(n_tr[i]) + 0e-14).tolist()  # What if there is an error...
    k_tf[i] = k_tr[0]
    k_ts[i] = k_tr[1]

#%%
# =============================================================================
# Calculation of Polarization Changes
# =============================================================================

    ang_refr_f = sp.acos(N_vec_inc.dot(S_tf[i]))
    ang_refr_s = sp.acos(N_vec_inc.dot(S_ts[i]))
    AOI = mpmath.degrees(sp.acos(kin[i].dot(N_vec_inc)))
    l_f = L/sp.cos(ang_refr_f)
    l_s = L/sp.cos(ang_refr_s)
    OPL_f = n_tr[i][0]*k_tf[i].dot(l_f*S_tf[i])
    OPL_s = n_tr[i][1]*k_ts[i].dot(l_s*S_ts[i])
    phase_jump = (l_f*S_tf[i] - l_s*S_ts[i]).dot(S_inc[i]) * 2*np.pi/WL


    t = sp.symbols('t')
    
    E_end[i] = Eend(E_inc[i], E_inc_s0[i], E_inc_p0[i], E_tf_t[i], E_ts_t[i], n_tr[i][0], n_tr[i][1], l_s, l_f, S_ts[i], S_tf[i], k_tf[i], k_ts[i], t, phase_jump)

#%%
# =============================================================================
# graphical presentation
# =============================================================================

from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

radius = 0.005
""" Sample Cylinder """
WedgeVector = normalize_np([0,0,1])
Xc, Yc, Zc, mask = data_for_cylinder_along_z(0, 0, radius, float(L), WedgeVector)
if np.all(WedgeVector == normalize_np([0,0,1])) == True:
    ax.plot_surface(Xc, Yc, Zc, color = 'b', alpha = 0.2, linewidth = 0, antialiased = True)
else:
    ax.scatter(Xc, Yc, Zc, color = 'b', alpha = 0.2, linewidth = 0, antialiased = True)
# ax.plot_trisurf(Xc.flatten(), Yc.flatten(), Z_masked.flatten(), cmap = mycmap, vmin = np.min(Zc), vmax = np.max(Zc), alpha = 0.5, linewidth = 0, antialiased = True)
ceiling = Circle((0, 0), radius, color = 'b', alpha = 0.2)
ceiling0 = Circle((0, 0), radius, color = 'b', alpha = 0.2)
ax.add_patch(ceiling)
ax.add_patch(ceiling0)
art3d.pathpatch_2d_to_3d(ceiling, z = L, zdir="z")
art3d.pathpatch_2d_to_3d(ceiling0, z = 0, zdir="z")       
""" Ray Tracing """
a = 0.1                 # Length-factor for the rays to be displayed
A = 1                   # Distance of Screen in Meter (Important for ray- and polarization analysis graph)
for i in kloc:
    b0 = InputBeamPoints[i-1][0]   # Offset in global-x
    b1 = InputBeamPoints[i-1][1]   # Offset in global-y
    b2 = InputBeamPoints[i-1][2]   # Offset in global-z
    a = np.linalg.norm(np.array([b0,b1,b2]))
    """ define exit-points on the sample and exit-ray's end-points """
    ExPoint_fx = l_f*S_tf[i][0] + b0
    ExPoint_fy = l_f*S_tf[i][1] + b1
    ExPoint_fz = l_f*S_tf[i][2]
    ExPoint_sx = l_s*S_ts[i][0] + b0
    ExPoint_sy = l_s*S_ts[i][1] + b1
    ExPoint_sz = l_s*S_ts[i][2]
    Pointer_fx = l_f*S_tf[i][0] + a*S_ts_f[i][0] + b0
    Pointer_fy = l_f*S_tf[i][1] + a*S_ts_f[i][1] + b1
    Pointer_fz = l_f*S_tf[i][2] + a*S_ts_f[i][2]
    Pointer_sx = l_s*S_ts[i][0] + a*S_ts_s[i][0] + b0
    Pointer_sy = l_s*S_ts[i][1] + a*S_ts_s[i][1] + b1
    Pointer_sz = l_s*S_ts[i][2] + a*S_ts_s[i][2]
    ax.plot([-a*kin[i][0] + b0, b0], [-a*kin[i][1] + b1, b1], [b2, 0], color = 'tab:blue')                # input beam direction
    ax.plot([b0, l_f*S_tf[i][0] + b0], [b1, l_f*S_tf[i][1] + b1], [0, l_f*S_tf[i][2]], color = 'blue')    # fast field direction
    ax.plot([b0, l_s*S_ts[i][0] + b0], [b1, l_s*S_ts[i][1] + b1], [0, l_s*S_ts[i][2]], color = 'magenta') # slow field direction
    ax.plot([ExPoint_fx, Pointer_fx], [ExPoint_fy, Pointer_fy], [ExPoint_fz, Pointer_fz], color = 'grey')                   # out-going field 
    ax.plot([ExPoint_sx, Pointer_sx], [ExPoint_sy, Pointer_sy], [ExPoint_sz, Pointer_sz], color = 'grey')                   # out-going field
    """ Save Screen-points """
    alpha = float (sp.acos(S_ts_f[i].dot(N_vec_out)))
    PlanePoints[i] = [np.add(np.array([ExPoint_fx, ExPoint_fy]), normalize(S_ts_f[i][0:2]) * np.tan(alpha) * A), np.add(np.array([ExPoint_sx, ExPoint_sy]), normalize(S_ts_s[i][0:2]) * np.tan(alpha) * A)]
    """ Crystal-axes orientation indicator """
    vec1 = ([ [0,0,0,radius,0,0], [0,0,0,0,radius,0], [0,0,0,0,0,L] ])
    for vector in vec1:
        v = np.array([vector[3], vector[4], vector[5]]).astype(np.float64)
        v = np.dot(RotMNz, np.dot(np.array(RotMNv).astype(np.float64), np.dot(np.array(RotMN2).astype(np.float64),v)))
        vlength = np.linalg.norm(v)
        ax.plot([vector[0], v[0]], [vector[1], v[1]], [vector[2], v[2]], linestyle = '--', color = 'k', linewidth = 1)
        # ax.quiver(vector[0],vector[1],vector[2],v[0],v[1],v[2], pivot='tail', length = vlength, linestyle = '--', arrow_length_ratio=0.01/vlength)
    """ Electric-field vector indicator """
    vec2 = ([ [b0, b1, b2, E_inc[i][0], E_inc[i][1], E_inc[i][2]] , [l_f/2*S_tf[i][0] + b0, l_f/2*S_tf[i][1] + b1, l_f/2*S_tf[i][2], E_tf[i][0], E_tf[i][1], E_tf[i][2]] , [l_s/2*S_ts[i][0] + b0, l_s/2*S_ts[i][1] + b1, l_s/2*S_ts[i][2], E_ts[i][0], E_ts[i][1], E_ts[i][2]]])
    for vector in vec2:
        v = np.array([vector[3], vector[4], vector[5]]).astype(np.float64)
        vlength = np.linalg.norm(v)
        color = 'tab:blue'
        if vector == vec2[1]:
            color = 'blue'
        if vector == vec2[2]:
            color = 'magenta'
        #ax.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5], pivot='tail', length=vlength/50, arrow_length_ratio=0.3/vlength, color = color)
ax.plot([0,L*local_N_vec[0]], [0,L*local_N_vec[1]], [0,L*local_N_vec[2]], color = 'k', linestyle='dotted') # axis combining both surfaces
ax.set_xlim(-radius, radius)
ax.set_ylim(-radius, radius)
ax.set_zlim(-0, float(L))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_box_aspect(( 1, 1, 1 ))

ax.view_init(elev=0., azim=60, vertical_axis = 'x')

plt.show()

# %%

from matplotlib.patches import Rectangle
import matplotlib as mpl

if len(kloc) == 1:

    fig, ax = plt.subplots(1,2,figsize=(12,5))
    fig.suptitle('(y,z) and (x,z) components of Poynting vectors')

    rect0 = Rectangle((-0.1,0),0.2,0.15, alpha = 0.2)
    rect1 = Rectangle((-0.1,0),0.2,0.15, alpha = 0.2)
    trans0 = mpl.transforms.Affine2D().rotate(-zeta) + ax[0].transData
    trans1 = mpl.transforms.Affine2D().rotate(-epsilon) + ax[1].transData
    rect0.set_transform(trans0)
    rect1.set_transform(trans1)
    v = np.array([vec1[2][3], vec1[2][4], vec1[2][5]])
    v = np.dot(RotMNz, np.dot(RotMNv, np.dot(RotMN2,v)))

    ax[0].set_title('(y,z)')
    ax[0].plot( [0, v[1]], [0, v[2]], color = 'k', linestyle = 'dotted', label='c-axis orientation')
    ax[0].legend()
    ax[0].add_patch(rect0)
    ax[0].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    # ax[0].plot([0,L*N_vec[1]], [0,L*N_vec[2]], linestyle='--')                                    # axis combining both surfaces
    inc_line = np.array(100 * N_vec_inc.cross( sp.Matrix([1,0,0]) )).astype(np.float64)
    ax[0].plot( [-inc_line[1],inc_line[1]] ,  [-inc_line[2],inc_line[2]], linestyle = '--', color = 'b')

    ax[0].plot( [-0.1*kin[i][1],0], [-0.1*kin[i][2],0], label='incoming beam')                            # input beam direction
    ax[0].plot( [0,l_f*S_tf[i][1]], [0,l_f*S_tf[i][2]], label='fast ray')               # ordinary field direction
    ax[0].plot( [0,l_s*S_ts[i][1]], [0,l_s*S_ts[i][2]], label='slow ray')          # extraordinary field direction
    ax[0].plot( [l_f*S_tf[i][1],l_f*S_tf[i][1] + 0.1*S_ts_f[i][1]], [l_f*S_tf[i][2],l_f*S_tf[i][2] + 0.1*S_ts_f[i][2]], color='tab:orange', linestyle='--', label='outgoing ray/o')                                      # out-going field ordinary
    ax[0].plot( [l_s*S_ts[i][1],l_s*S_ts[i][1] + 0.1*S_ts_s[i][1]], [l_s*S_ts[i][2],l_s*S_ts[i][2] + 0.1*S_ts_s[i][2]], color='tab:green', linestyle='--', label='outgoing ray/e')                                        # out-going field extraordinary
    

    ax[1].set_title('(x,z)')
    ax[1].plot( [0, v[0]], [0, v[2]], color = 'k', linestyle = 'dotted', label='c-axis orientation')
    ax[1].add_patch(rect1)
    ax[1].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    # ax[1].plot([0,L*N_vec[0]], [0,L*N_vec[2]], linestyle='--')                                      # axis combining both surfaces
    ax[1].plot( [-0.1*kin[i][0],0], [-0.1*kin[i][2],0])                                                   # input beam direction
    ax[1].plot( [0,l_f*S_tf[i][0]], [0,l_f*S_tf[i][2]])                                     # ordinary field direction
    ax[1].plot( [0,l_s*S_ts[i][0]], [0,l_s*S_ts[i][2]])                                     # extraordinary field direction
    ax[1].plot( [l_f*S_tf[i][0],l_f*S_tf[i][0] + 0.1*S_ts_f[i][0]], [l_f*S_tf[i][2],l_f*S_tf[i][2] + 0.1*S_ts_f[i][2]], color='tab:orange', linestyle='--')                                                              # out-going field ordinary 
    ax[1].plot( [l_s*S_ts[i][0],l_s*S_ts[i][0] + 0.1*S_ts_s[i][0]], [l_s*S_ts[i][2],l_s*S_ts[i][2] + 0.1*S_ts_s[i][2]], color='tab:green', linestyle='--')                                                              # out-going field extraordinary 

    ax[0].set_xlim((-0.01,0.01))
    ax[0].set_ylim((-0.1,0.2))
    ax[0].set_xlabel('y (m)')
    ax[0].set_ylabel('z (m)')
    ax[0].grid(True)
    ax[1].set_xlim((-0.01,0.01))
    ax[1].set_xlabel('x (m)')
    ax[1].set_ylabel('z (m)')
    ax[1].grid(True)
    plt.show()

# %%

from matplotlib.patches import Ellipse

if len(kloc) == 1:
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)

    rot_loc = sp.acos((RotMk.inv() * s_vec[i]).dot(sp.Matrix([1,0,0])))
    Rot_loc = R.as_matrix(R.from_euler('z', rot_loc.evalf(p)))
    Invert_Op = Rot_loc * RotMk.inv()
    Ax = [(Invert_Op * E_end[i][0]).subs(t,j).evalf(p) for j in np.arange(0, 1, 0.001)]
    Ax_inc = Invert_Op * E_inc[i]

# ax.plot(Ax[:,1], Ax[:,0],marker=None,color='b', label = 'transmitted field')
    a = max([Ax[j].norm() for j in range(len(Ax))])
    b = min([Ax[j].norm() for j in range(len(Ax))])
    pos_of_max = np.where(np.array([float(Ax[j].norm()) for j in range(len(Ax))]) == float(a))[0][0]
    rot = mpmath.degrees(sp.atan2(Ax[pos_of_max][0] , Ax[pos_of_max][1]))
    theta = mpmath.degrees(sp.atan2(max(Ax[:][1]), max(Ax[:][0])))

    ell = Ellipse((0,0), 2*float(a), 2*float(b), angle = float(rot), facecolor = 'none', edgecolor = 'b', lw = 2, label = 'transmitted field')
    print("Ellipticity: "+str(sp.sqrt(sp.Abs(a**2 - b**2)/a**2)))
    print("Rotation: "+str(rot.evalf(p))+" deg")
    ax.add_patch(ell)
    ax.plot( [Ax_inc[1], -Ax_inc[1]], [Ax_inc[0], -Ax_inc[0]], color='r', alpha = 0.5, label = 'incoming field')
    # ax.plot([0,np.max(Ax[:,1])] , [0,np.max(Ax[:,0])], color = 'k', linestyle = '--', label = 'transmitted field direction')

    # ax.set_title(r"$\xi$ (pol. angle) = "+str(round(theta,3))+" deg", fontsize = fontsize)
    ax.set_title(r"$\gamma$ (input polarization) = "+str(float(round(mpmath.degrees(theta_inc[i]),3)))+"$^\circ$", fontsize = fontsize + 2)
    ax.set_xlabel('E in p-direction')
    ax.tick_params(axis = 'x')
    ax.set_xlim(-1,1)
    ax.set_ylabel('E in s-direction')
    ax.tick_params(axis = 'y')
    ax.set_ylim(-1,1)
    ax.grid(True)
    ax.legend(loc = 'lower right')
    plt.show()

else:
    fig, ax = plt.subplots(1,1, figsize = (6,6))
    factor = 600
    
    for i in kloc:
        # rot_loc = np.arccos(np.dot(np.dot(np.linalg.inv(RotMk), E_inc[i]), np.array([1,0,0])))
        # Rot_loc = R.as_matrix(R.from_euler('z', rot_loc))
        # Invert_Op = np.dot(Rot_loc, np.linalg.inv(RotMk))
        # Ax = np.array([np.dot(Invert_Op, E_end[i][j][0]) for j in range(len(E_end[i]))])
        # Ax_inc = np.dot(Invert_Op, E_inc[i]) / factor
        Ax = np.array([E_end[i][j][0] for j in range(len(E_end[i]))])
        Ax_inc = E_inc[i] / factor

# ax.plot(Ax[:,1], Ax[:,0],marker=None,color='b', label = 'transmitted field')
        a = np.array([np.linalg.norm(Ax[j]) for j in range(len(Ax))]).max() / factor
        b = np.array([np.linalg.norm(Ax[j]) for j in range(len(Ax))]).min() / factor
        pos_of_max = np.where(np.array([np.linalg.norm(Ax[j]) for j in range(len(Ax))]) / factor == a)[0][0]
        rot = np.degrees(np.arctan2(Ax[pos_of_max, 0] , Ax[pos_of_max, 1]))
        theta = np.degrees(np.arctan2(np.max(Ax[:,1]), np.max(Ax[:,0])))
        print('θ = '+str(theta)+'°')
        center = np.add(PlanePoints[i][0], PlanePoints[i][1]) / 2
        
        ell = Ellipse((center[0], center[1]), 2*a, 2*b, angle = rot, facecolor = 'none', edgecolor = 'b', lw = 2, label = 'transmitted field', alpha = 0.5)
        ax.add_patch(ell)
        ax.plot( [center[0] + Ax_inc[1], center[0] - Ax_inc[1]], [center[1] + Ax_inc[0], center[1] - Ax_inc[0]], color='r', alpha = 0.5, label = 'incoming field')
        # ax.plot([center[0], center[0] + np.max(Ax[:,1])/factor] , [center[1], center[1] + np.max(Ax[:,0])/factor], color = 'k', linestyle = '--', label = 'transmitted field direction')
        
        ax.plot(PlanePoints[i][0][0], PlanePoints[i][0][1], linewidth = 0, marker = 'o', color = 'r')
        ax.plot(PlanePoints[i][1][0], PlanePoints[i][1][1], linewidth = 0, marker = 'o', color = 'b')
    
    ax.grid(True)
    ax.set_xlim(left = -0.005, right = 0.005)
    ax.set_ylim(bottom = -0.005, top = 0.005)
    ax.ticklabel_format(style = 'sci', scilimits = (0,0))
    ax.set_xlabel('y [m]')
    ax.set_ylabel('x [m]')
    
    plt.show()