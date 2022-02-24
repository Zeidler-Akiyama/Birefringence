#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:54:06 2022

@author: kyuubi
"""

import sympy as sp
import numpy as np
import pandas as pd

def RotMat(alpha,beta):
    return np.array([[np.cos(alpha),-np.sin(alpha)*np.sin(beta),np.sin(alpha)*np.cos(beta)],
                 [0,np.cos(beta),np.sin(beta)],
                 [-np.sin(alpha),-np.cos(alpha)*np.sin(beta),np.cos(beta)*np.cos(alpha)]])

def n_e(kin, eps1, eps2, N_vec, n_inc):
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
    # if opt == 'sympy':
    n_prime = sp.symbols('n_prime')
    k_in = sp.Matrix([sp.Rational(kin[0]), sp.Rational(kin[1]), sp.Rational(kin[2])])
    N_v = sp.Matrix([sp.Rational(N_vec[0]), sp.Rational(N_vec[1]), sp.Rational(N_vec[2])])
    # k_in = sp.Matrix([kin[0], kin[1], kin[2]])
    # N_v = sp.Matrix([N_vec[0], N_vec[1], N_vec[2]])
    
    if np.any(eps1-eps2) == False:
        k = n_inc*k_in + (-(n_inc*k_in.T*N_v)[0] - sp.sqrt((n_inc*k_in.T*N_v)[0]**2 + (n_prime**2 - n_inc**2)))*N_v
    else:
        k = n_inc*k_in + (-(n_inc*k_in.T*N_v)[0] + sp.sqrt((n_inc*k_in.T*N_v)[0]**2 + (n_prime**2 - n_inc**2)))*N_v
    
    eps2 = sp.Matrix([ [sp.nsimplify(eps2[0,0]), sp.nsimplify(eps2[0,1]), sp.nsimplify(eps2[0,2])], [sp.nsimplify(eps2[1,0]), sp.nsimplify(eps2[1,1]), sp.nsimplify(eps2[1,2])], [sp.nsimplify(eps2[2,0]), sp.nsimplify(eps2[2,1]), sp.nsimplify(eps2[2,2])] ])
    # k = sp.nsimplify(k) / sp.sqrt(sp.nsimplify(k[0]**2 + k[1]**2 + k[2]**2, tolerance=1e-15, rational=True))
    k = k/n_prime
    
    K = sp.Matrix([ [0, -k[2], k[1]] , [k[2], 0, -k[0]] , [-k[1], k[0], 0] ])
    
    M = eps2 + (n_prime*K)**2
    objective = sp.nsimplify( M.det())
    
    # try:
    #     sol = sp.solveset(objective, n_prime, domain = sp.S.Reals)
    #     if sol == sp.EmptySet:
    #         sol = sp.solve(objective, check = True)
    #         n_list = [float(sp.N(sp.re(sol[i]))) for i in range(len(sol)) if np.round(float(sp.N(sp.re(sol[i]))), 14)>=1.0]
    #     else:
    #         n_list = [float(sp.N(sp.re(sol.args[i]))) for i in range(len(sol.args)) if np.round(float(sp.N(sp.re(sol.args[i]))), 14)>=1.0]
    # except:
    #     sol = sp.solve(objective, check = True)
    #     n_list = [float(sp.N(sp.re(sol[i]))) for i in range(len(sol)) if np.round(float(sp.N(sp.re(sol[i]))), 14)>=1.0]
    
    sol = sp.solve(objective, check = True)
    n_list = [float(sp.N(sp.re(sol[i]))) for i in range(len(sol)) if np.round(float(sp.N(sp.re(sol[i]))), 14)>=1.0]
    n_list = list(dict.fromkeys(n_list))
    
    # sol = sp.solve(M.det(), check = True)
    # sol0 = [sp.N(sol[i]) for i in range(len(sol))]
    # n_list = np.real(np.array(sol).astype(np.complex64))
    # n_list = [float(sp.N(sp.re(sol[i]))) for i in range(len(sol)) if np.round(float(sp.N(sp.re(sol[i]))), 14)>=1.0]
    
    
    # if opt == 'scipy':
        
    #     def objective(n_prime):
    #         k_in = np.array([kin[0],kin[1],kin[2]])
    #         N_v = np.array([N_vec[0],N_vec[1],N_vec[2]])
    #         if np.any(eps1-eps2) == False:
    #             k = np.add(n_inc*k_in , (-n_inc*np.dot(k_in,N_v) - np.sqrt((n_inc*np.dot(k_in,N_v))**2 + (n_prime**2 - n_inc**2)))*N_v)
    #         else:
    #             k = np.add(n_inc*k_in , (-n_inc*np.dot(k_in,N_v) + np.sqrt((n_inc*np.dot(k_in,N_v))**2 + (n_prime**2 - n_inc**2)))*N_v)
    #         k = k/n_prime
    #         K = np.array([ [0, -k[2], k[1]] , [k[2], 0, -k[0]] , [-k[1], k[0], 0] ])
    #         M = np.add(eps2 , np.dot(n_prime*K, n_prime*K))
    #         return np.linalg.det(M)
        
    #     from scipy.optimize import root_scalar
        
    #     n_list = root_scalar(objective, method = 'secant', x0 = 1.753999, x1 = 1.7540001)
    
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
    
    n_list = np.array(n_list)
    if len(n_list)>1:
        # theta_f = np.arcsin(np.sin(AOI)/n_list[0])
        # theta_s = np.arcsin(np.sin(AOI)/n_list[1])
        k_tf = (n_inc*kin + np.multiply(-n_inc*np.dot(kin,N_vec) + np.sqrt(n_inc**2*np.dot(kin,N_vec)**2 + n_list[0]**2-n_inc**2),N_vec))
        k_tf = k_tf/np.linalg.norm(k_tf)
        k_ts = (n_inc*kin + np.multiply(-n_inc*np.dot(kin,N_vec) + np.sqrt(n_inc**2*np.dot(kin,N_vec)**2 + n_list[1]**2-n_inc**2),N_vec))
        k_ts = k_ts/np.linalg.norm(k_ts)
        k = [k_tf, k_ts]
    else:
        k_t = (n_inc*kin + np.multiply(-n_inc*np.dot(kin,N_vec) + np.sqrt(n_inc**2*np.dot(kin,N_vec)**2 + n_list**2-n_inc**2),N_vec))
        k_t = k_t/np.linalg.norm(k_t)
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
    n_list = np.array(n_list)
    if len(n_list)>1:
        k_rf = (n_inc*kin + np.multiply(-n_inc*np.dot(kin,N_vec) - np.sqrt(n_inc**2*np.dot(kin,N_vec)**2 + n_list[0]**2-n_inc**2),N_vec))
        k_rf = k_rf/np.linalg.norm(k_rf)
        k_rs = (n_inc*kin + np.multiply(-n_inc*np.dot(kin,N_vec) - np.sqrt(n_inc**2*np.dot(kin,N_vec)**2 + n_list[1]**2-n_inc**2),N_vec))
        k_rs = k_rs/np.linalg.norm(k_rs)
        k = [k_rf, k_rs]
    else:
        k_r = (n_inc*kin + np.multiply(-n_inc*np.dot(kin,N_vec) - np.sqrt(n_inc**2*np.dot(kin,N_vec)**2 + n_list**2-n_inc**2),N_vec))
        k_r = k_r/np.linalg.norm(k_r)
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
        K_f = np.array([[0,-kt[0][2],kt[0][1]],
                        [kt[0][2],0,-kt[0][0]],
                        [-kt[0][1],kt[0][0],0]])
        K_s = np.array([[0,-kt[1][2],kt[1][1]],
                        [kt[1][2],0,-kt[1][0]],
                        [-kt[1][1],kt[1][0],0]])
        K = np.add(eps , np.dot(n_list[0]*K_f , n_list[0]*K_f)), np.add(eps , np.dot(n_list[1]*K_s , n_list[1]*K_s)), K_f, K_s
    else:
        K_t = np.array([[0,-kt[0][2],kt[0][1]],
                        [kt[0][2],0,-kt[0][0]],
                        [-kt[0][1],kt[0][0],0]])
        K = np.add(eps , np.dot(n_list[0]*K_t , n_list[0]*K_t)), K_t
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
        K_f = np.array([[0,-kref[0][2],kref[0][1]],
                        [kref[0][2],0,-kref[0][0]],
                        [-kref[0][1],kref[0][0],0]])
        K_s = np.array([[0,-kref[1][2],kref[1][1]],
                        [kref[1][2],0,-kref[1][0]],
                        [-kref[1][1],kref[1][0],0]])
        K = np.add(eps , np.dot(n_list[0]*K_f , n_list[0]*K_f)), np.add(eps , np.dot(n_list[1]*K_s , n_list[1]*K_s)), K_f, K_s
    else:
        K_r = np.array([[0,-kref[0][2],kref[0][1]],
                        [kref[0][2],0,-kref[0][0]],
                        [-kref[0][1],kref[0][0],0]])
        K = np.add(eps , np.dot(n_list[0]*K_r , n_list[0]*K_r)), K_r
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
    E_inc = np.add(E_inc_s, E_inc_p)
    H_inc = np.add(H_inc_s, H_inc_p)
    # H_inc_p = np.dot(n_list[0]*K_inc, fast)#/np.linalg.norm(np.dot(ne[0]*K_inc, E_inc_p))
    # H_inc_s = np.dot(n_list[1]*K_inc, slow)#/np.linalg.norm(np.dot(ne[2]*K_inc, E_inc_s))
    C_s = np.array([np.dot(s1,E_inc_s),np.dot(s2,E_inc_s),np.dot(s1,H_inc_s),np.dot(s2,H_inc_s)])
    C_p = np.array([np.dot(s1,E_inc_p),np.dot(s2,E_inc_p),np.dot(s1,H_inc_p),np.dot(s2,H_inc_p)])
    C = np.array([np.dot(s1,E_inc),np.dot(s2,E_inc),np.dot(s1,H_inc),np.dot(s2,H_inc)])
    return C_s, C_p, C

def air_crystal(kin, eps1, eps2, N_vec, s_vec, p_vec, theta_inc, n_inc, I_inc, RotMk, k_Rot):
    
    n_tr = n_e(kin, eps1, eps2, N_vec, n_inc)
    if len(n_tr) == 1:
        n_tr = [n_tr[0], n_tr[0]]
    else:
        if n_tr[0] > n_tr[1]:
            n_tr = np.flip(n_tr, 0).tolist()
    # if np.linalg.norm(np.cross(kin, N_vec)) > 0:
    #     n_tr = list(dict.fromkeys(n_tr))
    #n_re = n_e(kin, eps1, eps1, N_vec, n_inc)
    n_re = [1.0] #list(dict.fromkeys(n_re))
    K_inc = np.array([[0,-kin[2],kin[1]], [kin[2],0,-kin[0]], [-kin[1],kin[0],0]])
    
    if bool(k_Rot) == True:
        E_inc_p0 = np.dot(RotMat(k_Rot[0], k_Rot[1]) , np.multiply(np.sin(theta_inc), np.array([0,1,0])))
        E_inc_s0 = np.dot(RotMat(k_Rot[0], k_Rot[1]) , np.multiply(np.cos(theta_inc), np.array([1,0,0])))
    else:
        E_inc_p0 = np.multiply(np.sin(theta_inc),p_vec)
        E_inc_s0 = np.multiply(np.cos(theta_inc),s_vec)
    E_inc = np.add(E_inc_p0,E_inc_s0)
    H_inc = np.dot(n_inc*K_inc,E_inc)
    H_inc_s = np.dot(n_inc*K_inc,E_inc_s0)
    H_inc_p = np.dot(n_inc*K_inc,E_inc_p0)
    S_inc = np.real(np.cross(E_inc,np.conj(H_inc)))/np.linalg.norm(np.real(np.cross(E_inc,np.conj(H_inc))))
    
    AOI = np.arccos(np.dot(-N_vec,S_inc))
    
    k_tr = k_t(n_inc,n_tr,AOI,kin,N_vec)
    k_re = k_r(n_inc,n_re,kin,N_vec)

    EMT = Eigen_matrix(k_tr,eps2,n_tr)
    EMR = Eigen_matrix_ref(k_re,eps1,n_re)
    
    if np.all(EMT[0]==EMT[1]) == True:
        SVD_tf = np.linalg.svd(EMT[0])
        SVD_ts = np.linalg.svd(EMT[1])
        E_tf = SVD_tf[0][:,2]
        E_ts = SVD_ts[0][:,1]
    else:
        SVD_tf = np.linalg.svd(EMT[0])
        SVD_ts = np.linalg.svd(EMT[1])
        if np.all(SVD_tf[0][:,2]==SVD_ts[0][:,2]) == True:
            E_tf = SVD_tf[0][:,2]
            E_ts = SVD_ts[0][:,1]
        else:    
            E_tf = SVD_tf[0][:,2]
            E_ts = SVD_ts[0][:,2]
    SVD_r = np.linalg.svd(EMR[0])
    E_rs = SVD_r[0][:,2]
    E_rp = SVD_r[0][:,1]
    if abs(np.dot(E_rs,s_vec)) < 1e-14:
        E_rs = s_vec
        E_rp = SVD_r[0][:,2]

    E_tf0 = E_tf[:]
    E_ts0 = E_ts[:]
    if len(EMT) == 2:
        H_tf = np.dot(n_tr[0]*EMT[1],E_tf)
        H_ts = np.dot(n_tr[0]*EMT[1],E_ts)
    else:
        H_tf = np.dot(n_tr[0]*EMT[2],E_tf)
        H_ts = np.dot(n_tr[1]*EMT[3],E_ts)
    H_rs = np.dot(n_re[0]*EMR[1],E_rs)
    H_rp = np.dot(n_re[0]*EMR[1],E_rp)

    S_tf = np.real(np.cross(E_tf,np.conj(H_tf)))/np.linalg.norm(np.real(np.cross(E_tf,np.conj(H_tf))))
    S_ts = np.real(np.cross(E_ts,np.conj(H_ts)))/np.linalg.norm(np.real(np.cross(E_ts,np.conj(H_ts))))
    S_rs = np.real(np.cross(E_rs,np.conj(H_rs)))/np.linalg.norm(np.real(np.cross(E_rs,np.conj(H_rs))))
    S_rp = np.real(np.cross(E_rp,np.conj(H_rp)))/np.linalg.norm(np.real(np.cross(E_rp,np.conj(H_rp))))
    S = [S_inc,S_tf, S_ts, S_rs, S_rp]

    if np.linalg.norm(np.cross(kin,N_vec)) == 0:
        s1 = np.dot(RotMk,np.array([1,0,0]))
    else:
        s1 = np.cross(kin,N_vec)
    s2 = np.cross(N_vec,s1)

    F = np.array([[np.dot(s1,E_tf), np.dot(s1,E_ts), np.dot(-s1,E_rs), np.dot(-s1,E_rp)],
                  [np.dot(s2,E_tf), np.dot(s2,E_ts), np.dot(-s2,E_rs), np.dot(-s2,E_rp)],
                  [np.dot(s1,H_tf), np.dot(s1,H_ts), np.dot(-s1,H_rs), np.dot(-s1,H_rp)],
                  [np.dot(s2,H_tf), np.dot(s2,H_ts), np.dot(-s2,H_rs), np.dot(-s2,H_rp)]])

    if np.linalg.norm(np.cross(S_inc,N_vec)) == 0:
        E_inc_s_norm = np.dot(RotMk,np.array([1,0,0]))
    else:
        E_inc_s_norm = np.cross(S_inc,N_vec)/np.linalg.norm(np.cross(S_inc,N_vec))
    E_inc_p_norm = np.cross(S_inc,E_inc_s_norm)
    H_inc_s_norm = np.dot(n_inc*K_inc, E_inc_s_norm)
    H_inc_p_norm = np.dot(n_inc*K_inc, E_inc_p_norm)
    
    A_s = np.dot(np.linalg.inv(F),C_vector(N_vec,s1,s2,K_inc,E_inc_s_norm,E_inc_p_norm,H_inc_s_norm,H_inc_p_norm)[0])
    A_f = np.dot(np.linalg.inv(F),C_vector(N_vec,s1,s2,K_inc,E_inc_s_norm,E_inc_p_norm,H_inc_s_norm,H_inc_p_norm)[1])

    a_stf = A_s[0]            # transmission coefficient to fast-ray
    a_sts = A_s[1]            # transmission coefficient to slow-ray
    a_srs = A_s[2]            # reflection coefficient to s-pol
    a_srp = A_s[3]            # reflection coefficient to p-pol

    a_ptf = A_f[0]            # transmission coefficient to fast-ray
    a_pts = A_f[1]            # transmission coefficient to slow-ray
    a_prs = A_f[2]            # reflection coefficient to s-pol
    a_prp = A_f[3]            # reflection coefficient to p-pol
    
    P_tf = np.dot(np.transpose(np.array([a_stf*E_tf, a_ptf*E_tf, S_tf])), (np.array([E_inc_s_norm, E_inc_p_norm, S_inc])))
    P_ts = np.dot(np.transpose(np.array([a_sts*E_ts, a_pts*E_ts, S_ts])), (np.array([E_inc_s_norm, E_inc_p_norm, S_inc])))
    # P_rs = np.dot(np.transpose(np.array([a_srs*E_rs, a_prs*E_rs, S_rs])), (np.array([E_inc_s_norm, E_inc_p_norm, S_inc])))
    # P_rp = np.dot(np.transpose(np.array([a_srp*E_rs, a_prp*E_rp, S_rp])), (np.array([E_inc_s_norm, E_inc_p_norm, S_inc])))
    P_r = np.dot(np.transpose(np.array([a_srs*E_rs, a_prp*E_rp, S_rs])), (np.array([E_inc_s_norm, E_inc_p_norm, S_inc])))

    E_t_f = np.dot(P_tf, E_inc)
    E_t_s = np.dot(P_ts, E_inc)
    # E_r_s = np.dot(P_rs, E_inc)
    # E_r_p = np.dot(P_rp, E_inc)
    E_r = np.dot(P_r, E_inc)
    
    if len(n_tr) > 1:
        I_t_f = n_tr[0]*np.dot(S_tf,N_vec)/n_inc/np.dot(S_inc,N_vec) * np.linalg.norm(E_t_f)**2 * I_inc / np.linalg.norm(E_inc+1e-14)**2
        I_t_s = n_tr[1]*np.dot(S_ts,N_vec)/n_inc/np.dot(S_inc,N_vec) * np.linalg.norm(E_t_s)**2 * I_inc / np.linalg.norm(E_inc+1e-14)**2
    else:
        I_t_f = n_tr[0]*np.dot(S_tf,N_vec)/n_inc/np.dot(S_inc,N_vec) * np.linalg.norm(E_t_f)**2 * I_inc / np.linalg.norm(E_inc+1e-14)**2
        I_t_s = n_tr[0]*np.dot(S_ts,N_vec)/n_inc/np.dot(S_inc,N_vec) * np.linalg.norm(E_t_s)**2 * I_inc / np.linalg.norm(E_inc+1e-14)**2
    # I_r_s = np.dot(S_rs,N_vec)/np.dot(S_inc,N_vec) * np.linalg.norm(E_r_s)**2 * I_inc / np.linalg.norm(E_inc)**2
    # I_r_p = np.dot(S_rp,N_vec)/np.dot(S_inc,N_vec) * np.linalg.norm(E_r_p)**2 * I_inc / np.linalg.norm(E_inc)**2
    I_r = np.dot(S_rs,N_vec)/np.dot(S_inc,N_vec) * np.linalg.norm(E_r)**2 * I_inc / np.linalg.norm(E_inc)**2
    I0 = [I_inc, I_t_f, I_t_s, I_r]

    df0 = pd.DataFrame([str(E_inc), str(n_tr), str(n_re), str(E_tf), str(E_ts), str(E_rs), str(E_rp), str(S[0]), str(S[1]), str(S[2]), str(S[3]), str(S[4]), str(E_t_f), str(E_t_s), str(E_r)] , ['incoming E-field','refractive index transmission', 'refractive index reflection', 'transmitted E-field direction (fast)', 'transmitted E-field direction (slow)', 'reflected E-field direction (s)', 'reflected E-field direction (p)','Poynting vector (incoming)','Poynting vector (fast)','Poynting vector (slow)','Poynting vector (s-reflected)','Poynting vector (p-reflected)','transmitted E-field (fast)', 'transmitted E-field (slow)','reflected E-field'], columns = [''])
    
    return n_tr, n_re, S, k_tr, k_re, I0, E_t_f, E_tf, E_t_s, E_ts, [E_inc, E_inc_s0, E_inc_p0], df0

def crystal_air_f(k_tr, eps1, eps2, N_vec, S_tf, n_inc, I_inc, E_t_f, E_tf, E_ts0, s_vec, p_vec, RotMk):
    
    n_tr_f = [1.0]#n_e(k_tr[0], eps2, eps1, N_vec, n_inc)
    n_tr_f = list(dict.fromkeys(n_tr_f))
    n_re_f = n_e(k_tr, eps2, eps2, N_vec, n_inc)

    AOI_f = np.arccos(np.dot(-N_vec,S_tf))
    k_tr_f = k_t(n_inc,n_tr_f,AOI_f,k_tr,N_vec)
    k_re_f = k_r(n_inc,n_re_f,k_tr,N_vec)

    EMT_f = Eigen_matrix(k_tr_f,eps1,n_tr_f)
    EMR_f = Eigen_matrix_ref(k_re_f,eps2,n_re_f)

    E_inc_f = E_t_f
    E_inc_s1 = np.cross(S_tf,E_t_f) / np.linalg.norm(np.cross(S_tf,E_t_f) + 1e-14)                   # Pseudo slow-field with no power
    H_inc_f = np.dot(n_inc*EMT_f[1],E_inc_f)
    H_inc_s = np.dot(n_inc*EMT_f[1],E_inc_s1)

    SVD_t = np.linalg.svd(EMT_f[0])
    if len(EMR_f) > 2:
        SVD_rs = np.linalg.svd(EMR_f[0])
        SVD_rp = np.linalg.svd(EMR_f[1])
    else:
        SVD_rs = np.linalg.svd(EMR_f[0])
        SVD_rp = np.linalg.svd(EMR_f[0])
    E_ts = SVD_t[0][:,2]
    E_tp = SVD_t[0][:,1]
    if abs(np.dot(E_ts,s_vec)) < 1e-14:
        E_ts = SVD_t[0][:,1]
        E_tp = SVD_t[0][:,2]

    if np.dot(SVD_rs[0][:,2],E_inc_f) < 1:
        E_rf0 = E_tf[:]
    else:
        E_rf0 = SVD_rs[0][:,2]
    if np.dot(SVD_rp[0][:,2],E_inc_s1) < 1:
        E_rs0 = E_ts0[:]
    else:
        E_rs0 = SVD_rp[0][:,2]

    H_ts = np.dot(n_tr_f[0]*EMT_f[1],E_ts)
    H_tp = np.dot(n_tr_f[0]*EMT_f[1],E_tp)
    if len(EMR_f) > 2:
        H_rf = np.dot(n_re_f[0]*EMR_f[2],E_rf0)
        H_rs = np.dot(n_re_f[1]*EMR_f[3],E_rs0)
    else:
        H_rf = np.dot(n_re_f[0]*EMR_f[1],E_rf0)
        H_rs = np.dot(n_re_f[0]*EMR_f[1],E_rs0)

    S_ts_f = np.real(np.cross(E_ts,np.conj(H_ts)))/np.linalg.norm(np.real(np.cross(E_ts,np.conj(H_ts))))
    S_tp_f = np.real(np.cross(E_tp,np.conj(H_tp)))/np.linalg.norm(np.real(np.cross(E_tp,np.conj(H_tp))))
    S_rf_f = np.real(np.cross(E_rf0,np.conj(H_rf)))/np.linalg.norm(np.real(np.cross(E_rf0,np.conj(H_rf))))
    S_rs_f = np.real(np.cross(E_rs0,np.conj(H_rs)))/np.linalg.norm(np.real(np.cross(E_rs0,np.conj(H_rs))))
    S = [S_tf,S_ts_f, S_tp_f, S_rf_f, S_rs_f]

    if np.linalg.norm(np.cross(k_tr,N_vec)) == 0:
        s1 = np.dot(RotMk,np.array([1,0,0]))
    else:
        s1 = np.cross(k_tr,N_vec)
    s2 = np.cross(N_vec,s1)

    F = np.array([[np.dot(s1,E_ts), np.dot(s1,E_tp), np.dot(-s1,E_rf0), np.dot(-s1,E_rs0)],
                  [np.dot(s2,E_ts), np.dot(s2,E_tp), np.dot(-s2,E_rf0), np.dot(-s2,E_rs0)],
                  [np.dot(s1,H_ts), np.dot(s1,H_tp), np.dot(-s1,H_rf), np.dot(-s1,H_rs)],
                  [np.dot(s2,H_ts), np.dot(s2,H_tp), np.dot(-s2,H_rf), np.dot(-s2,H_rs)]])

    E_inc_f_norm = E_inc_f/(np.linalg.norm(E_inc_f) + 1e-14)
    E_inc_s_norm = E_inc_s1/(np.linalg.norm(E_inc_s1) + 1e-14)
    # H_inc_f_norm = H_inc_f/(np.linalg.norm(H_inc_f) + 1e-14)
    # H_inc_s_norm = H_inc_s/(np.linalg.norm(H_inc_s) + 1e-14)
    H_inc_f_norm = np.dot(n_inc*EMT_f[1], E_inc_f_norm)
    H_inc_s_norm = np.dot(n_inc*EMT_f[1], E_inc_s_norm)


    A_f_f = np.dot(np.linalg.inv(F),C_vector(N_vec,s1,s2,EMT_f[1],E_inc_f_norm,E_inc_s_norm,H_inc_f_norm,H_inc_s_norm)[0])
    A_s_f = np.array([0,0,0,0])

    a_fts = A_f_f[0]            # transmission coefficient to s-pol
    a_ftp = A_f_f[1]            # transmission coefficient to p-pol
    a_frf = A_f_f[2]            # reflection coefficient to fast-ray
    a_frs = A_f_f[3]            # reflection coefficient to slow-ray

    a_sts = A_s_f[0]            # transmission coefficient to s-pol
    a_stp = A_s_f[1]            # transmission coefficient to p-pol
    a_srf = A_s_f[2]            # reflection coefficient to fast-ray
    a_srs = A_s_f[3]            # reflection coefficient to slow-ray

    P_t = np.dot(np.transpose(np.array([np.add(a_fts*E_ts,a_ftp*E_tp),np.zeros(3), S_ts_f])), (np.array([E_inc_f_norm, E_inc_s_norm, S_tf])))
    P_rf = np.dot(np.transpose(np.array([a_frf*E_rf0 ,np.zeros(3), S_rf_f])), (np.array([E_inc_f_norm, E_inc_s_norm, S_tf])))
    P_rs = np.dot(np.transpose(np.array([a_frs*E_rs0 ,np.zeros(3), S_rs_f])), (np.array([E_inc_f_norm, E_inc_s_norm, S_tf])))

    E_tf_t = np.dot(P_t, E_inc_f)
    E_rf_f = np.dot(P_rf, E_inc_f)
    E_rf_s = np.dot(P_rs, E_inc_f)
    
# E_tf_t = np.add(Fr0[0]*np.linalg.norm(E_inc_f_s)*s_vec , Fr0[1]*np.linalg.norm(E_inc_f_p)*p_vec)

    I_t = n_tr_f[0]*np.dot(S_ts_f,N_vec)/n_inc/np.dot(S_tf,N_vec) * np.linalg.norm(E_tf_t)**2 * I_inc / np.linalg.norm(E_inc_f+1e-14)**2
    I_r_f = np.dot(S_rf_f,N_vec)/np.dot(S_tf,N_vec) * np.linalg.norm(E_rf_f)**2 * I_inc / np.linalg.norm(E_inc_f + 1e-14)**2
    I_r_s = np.dot(S_rs_f,N_vec)/np.dot(S_tf,N_vec) * np.linalg.norm(E_rf_s)**2 * I_inc / np.linalg.norm(E_inc_f + 1e-14)**2
    I1 = [I_inc, I_t, I_r_f, I_r_s]

    df1 = pd.DataFrame([str(E_inc_f), str(n_tr_f), str(n_re_f), str(E_ts), str(E_tp), str(E_rf0), str(E_rs0), str(S[0]), str(S[1]), str(S[2]), str(S[3]), str(S[4]), str(E_tf_t),  str(E_rf_f), str(E_rf_s)] , ['E-field (fast)','refractive index transmission', 'refractive index reflection', 'transmitted E-field direction (s)', 'transmitted E-field direction (p)', 'reflected E-field direction (fast)', 'reflected E-field direction (slow)','Poynting vector (fast)','Poynting vector (s)','Poynting vector (p)','Poynting vector (fast-reflected)','Poynting vector (slow-reflected)','transmitted E-field','reflected E-field (fast)','reflected E-field (slow)'], columns = [''])

    return n_tr_f, n_re_f, S, k_tr_f, k_re_f, I1, E_ts, E_tp, [E_tf_t, E_rf_f, E_rf_s], df1

def crystal_air_s(k_tr, eps1, eps2, N_vec, S_ts, n_inc, I_inc, E_t_s, E_tf, E_ts0, s_vec, p_vec, RotMk):
    
    n_tr_s = [1.0]#n_e(k_tr[1], eps2, eps1, N_vec, n_inc)
    n_tr_s = list(dict.fromkeys(n_tr_s))
    n_re_s = n_e(k_tr, eps2, eps2, N_vec, n_inc)
    # if np.linalg.norm(np.cross(kin, N_vec)) > 0:
    #     n_re_s = list(dict.fromkeys(n_re_s))

    AOI_s = np.arccos(np.dot(-N_vec,S_ts))
    k_tr_s = k_t(n_inc,n_tr_s,AOI_s,k_tr,N_vec)
    k_re_s = k_r(n_inc,n_re_s,k_tr,N_vec)

    EMT_s = Eigen_matrix(k_tr_s,eps1,n_tr_s)
    EMR_s = Eigen_matrix_ref(k_re_s,eps2,n_re_s)

    E_inc_s = E_t_s
    E_inc_f1 = np.cross(S_ts,E_t_s) / (np.linalg.norm(np.cross(S_ts,E_t_s)) + 1e-14)                 # Pseudo fast-field with no power
    H_inc_s = np.dot(n_inc*EMT_s[1],E_inc_s)
    H_inc_f = np.dot(n_inc*EMT_s[1],E_inc_f1)

    SVD_t = np.linalg.svd(EMT_s[0])
    if len(EMR_s) > 2:
        SVD_rs = np.linalg.svd(EMR_s[0])
        SVD_rp = np.linalg.svd(EMR_s[1])
    else:
        SVD_rs = np.linalg.svd(EMR_s[0])
        SVD_rp = np.linalg.svd(EMR_s[0])
    E_ts = SVD_t[0][:,2]
    E_tp = SVD_t[0][:,1]
    if abs(np.dot(E_ts,s_vec)) < 1e-14:
        E_ts = SVD_t[0][:,1]
        E_tp = SVD_t[0][:,2]

    if np.dot(SVD_rs[0][:,2],E_inc_f1) < 1e-14:
        E_rf = E_tf
    else:
        E_rf = SVD_rs[0][:,2]
    if np.dot(SVD_rp[0][:,2],E_inc_s) < 1e-14:
        E_rs = E_ts0
    else:
        E_rs = SVD_rp[0][:,2]

    H_ts = np.dot(n_tr_s[0]*EMT_s[1],E_ts)
    H_tp = np.dot(n_tr_s[0]*EMT_s[1],E_tp)
    if len(EMR_s) > 2:
        H_rf = np.dot(n_re_s[0]*EMR_s[2],E_rf)
        H_rs = np.dot(n_re_s[1]*EMR_s[3],E_rs)
    else:
        H_rf = np.dot(n_re_s[0]*EMR_s[1],E_rf)
        H_rs = np.dot(n_re_s[0]*EMR_s[1],E_rs)

    S_ts_s = np.real(np.cross(E_ts,np.conj(H_ts)))/np.linalg.norm(np.real(np.cross(E_ts,np.conj(H_ts))))
    S_tp_s = np.real(np.cross(E_tp,np.conj(H_tp)))/np.linalg.norm(np.real(np.cross(E_tp,np.conj(H_tp))))
    S_rf_s = np.real(np.cross(E_rf,np.conj(H_rf)))/np.linalg.norm(np.real(np.cross(E_rf,np.conj(H_rf))))
    S_rs_s = np.real(np.cross(E_rs,np.conj(H_rs)))/np.linalg.norm(np.real(np.cross(E_rs,np.conj(H_rs))))
    S = [S_ts,S_ts_s, S_tp_s, S_rf_s, S_rs_s]

    if np.linalg.norm(np.cross(k_tr,N_vec)) == 0:
        s1 = np.dot(RotMk,np.array([1,0,0]))
    else:
        s1 = np.cross(k_tr,N_vec)
    s2 = np.cross(N_vec,s1)

    F = np.array([[np.dot(s1,E_ts), np.dot(s1,E_tp), np.dot(-s1,E_rf), np.dot(-s1,E_rs)],
                  [np.dot(s2,E_ts), np.dot(s2,E_tp), np.dot(-s2,E_rf), np.dot(-s2,E_rs)],
                  [np.dot(s1,H_ts), np.dot(s1,H_tp), np.dot(-s1,H_rf), np.dot(-s1,H_rs)],
                  [np.dot(s2,H_ts), np.dot(s2,H_tp), np.dot(-s2,H_rf), np.dot(-s2,H_rs)]])

    E_inc_s_norm = E_inc_s/(np.linalg.norm(E_inc_s) + 1e-14)
    E_inc_f_norm = E_inc_f1/(np.linalg.norm(E_inc_f1) + 1e-14)
    # H_inc_s_norm = H_inc_s/(np.linalg.norm(H_inc_s) + 1e-14)
    # H_inc_f_norm = H_inc_f/(np.linalg.norm(H_inc_f) + 1e-14)
    H_inc_f_norm = np.dot(n_inc*EMT_s[1], E_inc_f_norm)
    H_inc_s_norm = np.dot(n_inc*EMT_s[1], E_inc_s_norm)


# s_vec_s = np.cross(S_ts,N_vec) / np.linalg.norm(np.cross(S_ts,N_vec))
# alpha = np.arccos(np.dot(E_inc_f/np.linalg.norm(E_inc_s + 1e-14),s_vec_s))
# E_inc_s_s = np.cos(alpha)*s_vec_s * np.linalg.norm(E_inc_s)
# E_inc_s_p = np.sin(alpha)*np.cross(s_vec_s, S_ts) * np.linalg.norm(E_inc_s)

# Fr1 = Fresnel(n_inc, n_tr_s[0], S_ts, S_ts_s, N_vec)
# if np.dot(E_inc_s,s_vec) < 1e-14:
#     A_s_s = [0,Fr1[1],Fr1[3],0]
# else:
#     A_s_s = [Fr1[0],0,0,Fr1[2]]
  
    A_s_s = np.dot(np.linalg.inv(F),C_vector(N_vec,s1,s2,EMT_s[1],E_inc_f_norm,E_inc_s_norm,H_inc_f_norm,H_inc_s_norm)[1])
    A_s_f = np.array([0,0,0,0])

    a_fts = A_s_f[0]            # transmission coefficient to s-pol
    a_ftp = A_s_f[1]            # transmission coefficient to p-pol
    a_frf = A_s_f[2]            # reflection coefficient to fast-ray
    a_frs = A_s_f[3]            # reflection coefficient to slow-ray

    a_sts = A_s_s[0]            # transmission coefficient to s-pol
    a_stp = A_s_s[1]            # transmission coefficient to p-pol
    a_srf = A_s_s[2]            # reflection coefficient to fast-ray
    a_srs = A_s_s[3]            # reflection coefficient to slow-ray

    P_t = np.dot(np.transpose(np.array([np.add(a_sts*E_ts,a_stp*E_tp),np.zeros(3), S_ts_s])), (np.array([E_inc_s_norm, E_inc_f_norm,    S_ts])))
    P_rf = np.dot(np.transpose(np.array([a_srf*E_rf ,np.zeros(3), S_rf_s])), (np.array([E_inc_s_norm, E_inc_f_norm, S_ts])))
    P_rs = np.dot(np.transpose(np.array([a_srs*E_rs ,np.zeros(3), S_rs_s])), (np.array([E_inc_s_norm, E_inc_f_norm, S_ts])))

    E_ts_t = np.dot(P_t, E_inc_s)
    E_rs_f = np.dot(P_rf, E_inc_s)
    E_rs_s = np.dot(P_rs, E_inc_s)

# E_ts_s = np.add(Fr1[0]*np.linalg.norm(E_inc_s_s)*s_vec , Fr1[1]*np.linalg.norm(E_inc_s_p)*p_vec)

    I_t = n_tr_s[0]*np.dot(S_ts_s,N_vec)/n_inc/np.dot(S_ts,N_vec) * np.linalg.norm(E_ts_t)**2 * I_inc / np.linalg.norm(E_inc_s+1e-14)**2
    I_r_f = np.dot(S_rf_s,N_vec)/np.dot(S_ts,N_vec) * np.linalg.norm(E_rs_f)**2 * I_inc / np.linalg.norm(E_inc_s + 1e-14)**2
    I_r_s = np.dot(S_rs_s,N_vec)/np.dot(S_ts,N_vec) * np.linalg.norm(E_rs_s)**2 * I_inc / np.linalg.norm(E_inc_s + 1e-14)**2
    I2 = [I_inc, I_t, I_r_f, I_r_s]

    df2 = pd.DataFrame([str(E_inc_s), str(n_tr_s), str(n_re_s), str(E_ts), str(E_tp), str(E_rf), str(E_rs), str(S[0]), str(S[1]), str(S[2]), str(S[3]), str(S[4]), str(E_ts_t),  str(E_rs_f), str(E_rs_s)] , ['E-field (slow)','refractive index transmission', 'refractive index reflection', 'transmitted E-field direction (s)', 'transmitted E-field direction (p)', 'reflected E-field direction (fast)', 'reflected E-field direction (slow)','Poynting vector (fast)','Poynting vector (s)','Poynting vector (p)','Poynting vector (fast-reflected)','Poynting vector (slow-reflected)','transmitted E-field','reflected E-field (fast)','reflected E-field (slow)'], columns = [''])
    
    return n_tr_s, n_re_s, S, k_tr_s, k_re_s, I2, E_ts, E_tp, [E_ts_t, E_rs_f, E_rs_s], df2
