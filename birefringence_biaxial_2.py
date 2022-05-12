#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:44:47 2020

@author: kyuubi
"""

"""
Calculating the exptected birefringence from a ray passing through a biaxial crystal from an arbitrary direction. Biaxial is meant to be in a general formalism. So, this code should also represent the case of a uniaxial crystal in which we have strain in a certain direction, thus altering the index ellipsoid.
It is assumed that the crystal parameters do not change throuout the beam path (refer to "birefringence_biaxial_3.py")!

k --> wavevector

The preferred direction is toward the z-axis, meaning that the surface and the incident k-vector are oriented in z-direction per default

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R
from birefringence_modules import RotMat, air_crystal, crystal_air_f, crystal_air_s

plt.close('all')
fontsize = 14
plt.rcParams["font.size"] = fontsize
plt.rcParams.update({
    "text.usetex" : True,
    "font.family" : "Helvetica"
    })

#%%
# =============================================================================
# Functions
# =============================================================================

def Eend(E_inc, E_inc_s0, E_inc_p0, E_tf_t, E_ts_t, n_low, n_high, l_s, l_f, S_ts, S_tf, k_tr_f, k_tr_s, t, phase_jump):
    E_t_f_time = np.real(np.linalg.norm(E_inc) * E_tf_t*np.exp(1j*2*np.pi/WL*n_low*np.dot(k_tr_f,l_f*S_tf)) * np.exp(-1j*t*2*np.pi))
    E_t_s_time = np.real(np.linalg.norm(E_inc) * E_ts_t*np.exp(1j*2*np.pi/WL*n_high*np.dot(k_tr_s,l_s*S_ts)) * np.exp(-1j*t*2*np.pi) * np.exp(1j*phase_jump))
    E_inc_s_time = np.real(E_inc_s0 * np.exp(-1j*t*2*np.pi))
    E_inc_p_time = np.real(E_inc_p0 * np.exp(-1j*t*2*np.pi))
    return np.add(E_t_f_time, E_t_s_time), E_t_f_time, E_t_s_time, E_inc_s_time, E_inc_p_time

def normalize(x):
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
    z_incl = 2*radius * np.sin(abs(np.pi/2 - np.arccos(np.dot(WedgeVector, normalize([WedgeVector[0],WedgeVector[1],0])))))
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
    x_R = np.pi*w0**2*1/w
    Rx = x*(1 + (x_R/x)**2)
    wx = w0*np.sqrt(1 + (x/x_R)**2)
    # r = {}
    r0 = []
    for i in range(h):
        # r0 = []
        if i == 0:
            if isinstance(x, np.ndarray) == True:
                z = np.zeros(len(x)); y = np.zeros(len(x))
            else:
                z = 0; y = 0
            r0.append([z + offset[0], y + offset[1], x + offset[2]])
        else:
            for j in range(rot):
                if np.any(np.isnan(Rx)) == True:
                    z = (i * wx/h) * np.cos(j*2*np.pi/rot)
                    y = (i * wx/h) * np.sin(j*2*np.pi/rot)
                else:
                    alpha_max = 0.5 * np.arcsin(2 * wx/Rx)
                    z = np.sin(i * alpha_max/h) * Rx * np.cos(j*2*np.pi/rot)
                    y = np.sin(i * alpha_max/h) * Rx * np.sin(j*2*np.pi/rot)
                    x = x + Rx - (np.cos(i * alpha_max/h) * Rx)
                r0.append([z + offset[0], y + offset[1], x + offset[2]])
    #     r[i] = np.array(r0[i])
    return r0

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
I = {}
E_end = {}
PlanePoints = {}

#%%
# =============================================================================
# Orientation of the crystal
# =============================================================================

local_N_vec = np.array([0,0,1])    # Normal Vector of surface in global coordinates
center = [0,0,0.075]                # center-point of the crystal (dimensions in meter)
L = 2*center[2]                     # length of the crystal
nSa = 1.754#-1.5e-6                         # refractive index of Fast-axis of biaxial medium
nMa = 1.754 - 2.8e-6                 # refractive index of Medium-axis of biaxial medium
nFa = 1.747                         # refractive index of Slow-axis of biaxial medium

#%%
# =============================================================================
# Defining incoming beam and orientation
# =============================================================================
"""angle of k in xz plane (0 --> z) """
alpha = np.radians(0.0)  
"""angle of k from xz plane to y (0 --> xz)"""                                                                
beta =  np.radians(2.0)#np.arcsin(np.sin(np.radians(3/4))*nMa) - np.radians(0)
""""angle of Ca in medium 1 in xz plane (0 --> z)"""
gamma1 = np.radians(0.0)
""""angle of Ca in medium 1 from xz plane to y (0 --> xz)"""
delta1 = np.radians(0.0) 
""""angle of Ca in medium 2 in xz plane (0 --> z) """
gamma2 = np.radians(0.0)
""""angle of Ca in medium 2 from xz plane to y (0 --> xz)"""
delta2 = np.radians(0.0)
""""angle of N_vec in xz plane"""
epsilon = np.radians(0.0)
""""angle of N_vec from xz plane to y"""
zeta = np.radians(0.0)

WL = 1.064e-6
w = 2*np.pi*299792458/WL
InputBeamPoints = wavefront_coor(1, 1, WL, 0.00005, 0, -0.5, [0, 0, 0.4])
dInputBeamPoints = wavefront_coor(1, 1, WL, 0.00005, 0, 0.01, [0, 0, 0.4])
for i in range(len(InputBeamPoints)):
    kloc[i+1] = normalize(np.add( np.array(dInputBeamPoints[i]) , -1*np.array(InputBeamPoints[i]) ))
# kloc[1] = normalize([0,0,1])
# kloc[2] = normalize([-0.05,0,1])

RotMk = RotMat(alpha, beta)
RotMN1 = RotMat(gamma1-epsilon,delta1-zeta)
RotMN2 = RotMat(gamma2-epsilon,delta2-zeta)
RotMNv = RotMat(epsilon,zeta)
RotMNz = R.as_matrix(R.from_euler('z', 20, degrees = True))

eps2 = np.array([[nSa**2,0,0], [0,nMa**2,0], [0,0,nFa**2]])
eps_sap = np.dot(RotMNv, np.dot(np.dot(RotMN2,np.dot(eps2,np.transpose(RotMN2))), np.transpose(RotMNv)))
eps2 = np.dot(RotMNz, np.dot(eps_sap, np.transpose(RotMNz)))

eps1 = np.array([[1**2,0,0], [0,1**2,0], [0,0,1**2]])
eps_air = np.dot(RotMNv, np.dot(np.dot(RotMN1,np.dot(eps1,np.transpose(RotMN1))), np.transpose(RotMNv)))

for i in kloc:

    N_vec_inc = np.dot(RotMNv,local_N_vec)
    N_vec_out = local_N_vec.copy()
    kin[i] = np.dot(RotMk,kloc[i])

    if np.linalg.norm(np.cross(N_vec_inc,kin[i])) == 0:
        s_vec[i] = np.dot(RotMk,np.array([1,0,0]))
    else:
        s_vec[i] = normalize(np.cross(kin[i],N_vec_inc))
    p_vec = np.cross(s_vec[i],kin[i])

for i in kloc:
    if len(kloc) > 1:
        theta_inc[i] = 0#-np.arccos(np.dot(s_vec[1], s_vec[i]))
        k_Rot_beta = np.arccos(np.dot(np.array([0,0,1]) , normalize([0 , kin[i][1] , kin[i][2]])))
        k_Rot_alpha = np.arccos(np.dot(normalize([0, kin[i][1] , kin[i][2]]), kin[i]))
        if kin[i][0] < 0:
            k_Rot_alpha = -k_Rot_alpha
        if kin[i][1] < 0:
            k_Rot_beta = -k_Rot_beta
        k_Rot[i] = [k_Rot_alpha, k_Rot_beta]
    else:
        theta_inc[i] = np.radians(0)
        k_Rot[i] = [alpha, beta]
    K_inc = np.array([[0,-kin[i][2],kin[i][1]], [kin[i][2],0,-kin[i][0]], [-kin[i][1],kin[i][0],0]])

#%%
# =============================================================================
# Running code
# =============================================================================

    n_inc = 1.0
    I_inc = 1.0
    
    F0 = air_crystal(kin[i], eps_air, eps2, N_vec_inc, s_vec[i], p_vec, theta_inc[i], n_inc, I_inc, RotMk, k_Rot[i])
    
    F1 = crystal_air_f(F0[3][0], eps_air, eps2, N_vec_out, F0[2][1], F0[0][0], F0[5][1], F0[6], F0[7], F0[9], s_vec[i], p_vec, RotMk) 
    
    F2 = crystal_air_s(F0[3][1], eps_air, eps2, N_vec_out, F0[2][2], F0[0][1], F0[5][2], F0[8], F0[7], F0[9], s_vec[i], p_vec, RotMk)

    n_tr[i] = F0[0]
    n_tr[i] = (np.array(n_tr[i]) + 0e-14).tolist()  # What if there is an error...
    S_inc[i] = F0[2][0]
    S_tf[i] = F0[2][1]
    S_ts[i] = F0[2][2]
    S_ts_f[i] = F1[2][1]
    S_ts_s[i] = F2[2][1]
    k_tf[i] = F0[3][0]
    k_ts[i] = F0[3][1]
    E_inc[i] = F0[10][0]
    E_inc_s0[i] = F0[10][1]
    E_inc_p0[i] = F0[10][2]
    E_tf_t[i] = F1[8][0]
    E_ts_t[i] = F2[8][0]
    E_tf[i] = F0[7]
    E_ts[i] = F0[9]
    
    I[i] = [F0[5], F1[5], F2[5]]
    
    df3 = pd.DataFrame([I[i][0][0], I[i][1][1] + I[i][2][1], I[i][0][3] + I[i][1][2] + I[i][2][2] + I[i][1][3] + I[i][2][3]], ['incoming Intensity', 'transmitted Intensity', 'reflected Intensity'], columns=[''])
    print('')
    print('Intensities:')
    print(df3)
    print()

#%%
# =============================================================================
# Calculation of Polarization Changes
# =============================================================================

    ang_refr_f = np.arccos(np.dot(N_vec_inc, S_tf[i]))
    ang_refr_s = np.arccos(np.dot(N_vec_inc, S_ts[i]))
    AOI = np.degrees(np.arccos(np.dot(kin[i], N_vec_inc)))
    l_f = L/np.cos(ang_refr_f)
    l_s = L/np.cos(ang_refr_s)
    OPL_f = n_tr[i][0]*np.dot(k_tf[i], l_f*S_tf[i])
    OPL_s = n_tr[i][1]*np.dot(k_ts[i], l_s*S_ts[i])
    phase_jump = np.dot((l_f*S_tf[i] - l_s*S_ts[i]), S_inc[i]) * 2*np.pi/WL


    t = np.arange(0, 1, 0.001)
    Eend0 = []
    # angle = []
    for j in range(len(t)):
        Eend0.append(Eend(E_inc[i], E_inc_s0[i], E_inc_p0[i], E_tf_t[i], E_ts_t[i], n_tr[i][0], n_tr[i][1], l_s, l_f, S_ts[i], S_tf[i], k_tf[i], k_ts[i], t[j], phase_jump))
        # angle.append(np.arccos(np.dot(Eend0[i][1], Eend0[i][2])/np.linalg.norm(Eend0[i][1])/np.linalg.norm(Eend0[i][2])))
    # angle = np.array(angle)
    Eend0 = np.array(Eend0)
    E_end[i] = Eend0

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
WedgeVector = normalize([0,0,1])
Xc, Yc, Zc, mask = data_for_cylinder_along_z(0, 0, radius, L, WedgeVector)
if np.all(WedgeVector == normalize([0,0,1])) == True:
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
    alpha = np.arccos(np.dot(S_ts_f[i], N_vec_out))
    PlanePoints[i] = [np.add(np.array([ExPoint_fx, ExPoint_fy]), normalize(S_ts_f[i][0:2]) * np.tan(alpha) * A), np.add(np.array([ExPoint_sx, ExPoint_sy]), normalize(S_ts_s[i][0:2]) * np.tan(alpha) * A)]
    """ Crystal-axes orientation indicator """
    vec1 = ([ [0,0,0,radius,0,0], [0,0,0,0,radius,0], [0,0,0,0,0,L] ])
    for vector in vec1:
        v = np.array([vector[3], vector[4], vector[5]])
        v = np.dot(RotMNz, np.dot(RotMNv, np.dot(RotMN2,v)))
        vlength = np.linalg.norm(v)
        ax.plot([vector[0], v[0]], [vector[1], v[1]], [vector[2], v[2]], linestyle = '--', color = 'k', linewidth = 1)
        # ax.quiver(vector[0],vector[1],vector[2],v[0],v[1],v[2], pivot='tail', length = vlength, linestyle = '--', arrow_length_ratio=0.01/vlength)
    """ Electric-field vector indicator """
    vec2 = ([ [b0, b1, b2, E_inc[i][0], E_inc[i][1], E_inc[i][2]] , [l_f/2*S_tf[i][0] + b0, l_f/2*S_tf[i][1] + b1, l_f/2*S_tf[i][2], E_tf[i][0], E_tf[i][1], E_tf[i][2]] , [l_s/2*S_ts[i][0] + b0, l_s/2*S_ts[i][1] + b1, l_s/2*S_ts[i][2], E_ts[i][0], E_ts[i][1], E_ts[i][2]]])
    for vector in vec2:
        v = np.array([vector[3], vector[4], vector[5]])
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
ax.set_zlim(-0, L)
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
    inc_line = 100 * np.cross( N_vec_inc , np.array([1,0,0]) )
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

    rot_loc = np.arccos(np.dot(np.dot(np.linalg.inv(RotMk), s_vec[i]), np.array([1,0,0])))
    Rot_loc = R.as_matrix(R.from_euler('z', rot_loc))
    Invert_Op = np.dot(Rot_loc, np.linalg.inv(RotMk))
    Ax = np.array([np.dot(Invert_Op, E_end[i][j][0]) for j in range(len(E_end[i]))])
    Ax_inc = np.dot(Invert_Op, E_inc[i])

# ax.plot(Ax[:,1], Ax[:,0],marker=None,color='b', label = 'transmitted field')
    a = np.array([np.linalg.norm(Ax[j]) for j in range(len(Ax))]).max()
    b = np.array([np.linalg.norm(Ax[j]) for j in range(len(Ax))]).min()
    pos_of_max = np.where(np.array([np.linalg.norm(Ax[j]) for j in range(len(Ax))]) == a)[0][0]
    rot = np.degrees(np.arctan2(Ax[pos_of_max, 0] , Ax[pos_of_max, 1]))
    theta = np.degrees(np.arctan2(np.max(Ax[:,1]), np.max(Ax[:,0])))

    ell = Ellipse((0,0), 2*a, 2*b, angle = rot, facecolor = 'none', edgecolor = 'b', lw = 2, label = 'transmitted field')
    print("Ellipticity: "+str(np.sqrt(abs(a**2 - b**2)/a**2)))
    print("Rotation: "+str(rot)+" deg")
    ax.add_patch(ell)
    ax.plot( [Ax_inc[1], -Ax_inc[1]], [Ax_inc[0], -Ax_inc[0]], color='r', alpha = 0.5, label = 'incoming field')
    # ax.plot([0,np.max(Ax[:,1])] , [0,np.max(Ax[:,0])], color = 'k', linestyle = '--', label = 'transmitted field direction')

    # ax.set_title(r"$\xi$ (pol. angle) = "+str(round(theta,3))+" deg", fontsize = fontsize)
    ax.set_title(r"$\gamma$ (input polarization) = "+str(round(np.degrees(theta_inc[i]),3))+"$^\circ$", fontsize = fontsize + 2)
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