import numpy as np
from numpy import sqrt, array, exp 
from numpy.random import multivariate_normal, choice

from scipy.linalg import expm

import qutip as qt
from qutip import *
import matplotlib.pyplot as plt

def ibm_sherbrooke_params():
    T1 = 3.73e-04 
    T2 = 3.36e-04
    p = 2.31e-04
    tg = 5.33333333e-07
    return p, T1, T2, tg



def parameters(pp, pT1, pT2, ptg):
    p = pp                  # depolarizing probability
    T1 = pT1                 # relaxation time (seconds)
    T2 = pT2                 # dephasing time (seconds)
    tg = ptg                 # gate time (seconds)

    # Derived noise amplitudes
    ed = sqrt(p / 4)
    e1 = sqrt(tg / T1) if T1 else 0
    e2 = sqrt(tg / T2) if T2 else 0
    ep = sqrt(0.5 * (e2**2 - e1**2 / 2)) if T2 else 0
    return ed, e1, e2, ep

def compute_R(psi, phi):
    X = Qobj([[0, 1], [1, 0]])
    Y = Qobj([[0, -1j], [1j, 0]])
    Z = Qobj([[1, 0], [0, -1]])
    # === Rotation Axes ===
    R = (
        np.sin(psi) * np.cos(phi) * X +
        np.sin(psi) * np.sin(phi) * Y +
        np.cos(psi) * Z
    )
    return R

def make_U(theta,phi, psi):
    I = qeye(2)

    R = compute_R(psi, phi)
    alpha = theta / 2
    U = np.cos(alpha) * I - 1j * np.sin(alpha) * R
    return U



#######  X - Gate    #######

def X_gate_depolarization_noise_x(theta, ed):
    # Variances and covariances for depolarization Itô processes (X axis)
    Var_X = 1
    Var_Y = 0
    Var_Z = 0
    Cov_X_Y = 0
    Cov_X_Z = 0
    Cov_Y_Z = 0

    mean = array([0, 0, 0])
    cov = array([
        [Var_X, Cov_X_Y, Cov_X_Z],
        [Cov_X_Y, Var_Y, Cov_Y_Z],
        [Cov_X_Z, Cov_Y_Z, Var_Z]
    ])

    # Sample one realization: [dX, dY, dZ]
    sample_X = np.random.multivariate_normal(mean, cov,1)
    dX, dY, dZ = sample_X[0]
    
    # Construct Hermitian matrix: Id = dX·σ_X + dY·σ_Y + dZ·σ_Z
    Idx = ed * array([
        [dZ, dX - 1j * dY],
        [dX + 1j * dY, -dZ]
    ])
    
    return Idx


def X_gate_depolarization_noise_y(theta, ed):
    """Simulates Hermitian noise matrix for depolarization along the Y axis."""
    # Explicit variances and covariances
    Var_X = 0
    Var_Y = (2 * theta + np.sin(2 * theta)) / (4 * theta)
    Var_Z = (2 * theta - np.sin(2 * theta)) / (4 * theta)

    Cov_X_Y = 0
    Cov_X_Z = 0
    Cov_Y_Z = (-1*np.sin(theta) ** 2) / (2 * theta)

    mean = np.array([0, 0, 0])
    cov = np.array([
        [Var_X,  Cov_X_Y,  Cov_X_Z],
        [Cov_X_Y, Var_Y,   Cov_Y_Z],
        [Cov_X_Z, Cov_Y_Z,  Var_Z]
    ])

    # Sample one realization: [dX, dY, dZ]
    sample_Y = np.random.multivariate_normal(mean, cov,1)
    dX, dY, dZ = sample_Y[0]
    
    # Construct Hermitian matrix: Id = dX·σ_X + dY·σ_Y + dZ·σ_Z
    Idy = ed * array([
        [dZ, dX - 1j * dY],
        [dX + 1j * dY, -dZ]
    ])

    return Idy


def X_gate_depolarization_noise_z(theta, ed):
    """Simulates Hermitian noise matrix for depolarization along the Y axis."""
    # Explicit variances and covariances
    Var_X = 0
    Var_Y = (2 * theta - np.sin(2 * theta)) / (4 * theta)
    Var_Z = (2 * theta + np.sin(2 * theta)) / (4 * theta)

    Cov_X_Y = 0
    Cov_X_Z = 0
    Cov_Y_Z = (np.sin(theta) ** 2) / (2 * theta)

    mean = np.array([0, 0, 0])
    cov = np.array([
        [Var_X,  Cov_X_Y,  Cov_X_Z],
        [Cov_X_Y, Var_Y,   Cov_Y_Z],
        [Cov_X_Z, Cov_Y_Z,  Var_Z]
    ])

    # Sample one realization: [dX, dY, dZ]
    sample_Z = np.random.multivariate_normal(mean, cov,1)
    dX, dY, dZ = sample_Z[0]

    # Construct Hermitian matrix: Id = dX·σ_X + dY·σ_Y + dZ·σ_Z
    Idz = ed * array([
        [dZ, dX - 1j * dY],
        [dX + 1j * dY, -dZ]
    ])

    return Idz


def X_gate_deterministic_noise(theta, phi,e1):
    # 3 continued) Deterministic relaxation contribution
    det1 = (theta - np.sin(theta)) / (2 * theta)
    det2 = 1j*(np.cos(theta)-1) / (2*theta)
    det3 = (theta + np.sin(theta)) / (2 * theta)

    deterministic = -e1**2 / 2 * array([
        [det1, 1j/2 * exp(-1j * phi) * det2],
        [-1j/2 * exp(1j * phi) * det2, det3]
    ])
    return deterministic


def X_gate_construction(para_p, para_T1, para_T2, para_tg, additional_noise_bool = False):
    theta = np.pi  # Arbitrary phase, can be set to pi/2
    phi = 0  # X gate is a rotation of pi/2 around the X axis
    psi = np.pi / 2  # X gate is a rotation of pi/2 around the X axis 
    ed, e1, e2, ep = parameters(para_p, para_T1, para_T2, para_tg)

    U = make_U(theta,phi,psi)

    Idx = X_gate_depolarization_noise_x(theta,ed)
    Idy = X_gate_depolarization_noise_y(theta,ed)
    Idz = X_gate_depolarization_noise_z(theta,ed)
    
    if not additional_noise_bool:
        result = U @ expm(1j * (Idx + Idy + Idz))
        
    else:
        Ir = e1*(Idx - 1j*Idy)/(ed*2)
        Ip = ep*Idz/ed
        deterministic = X_gate_deterministic_noise(theta, phi, e1)
        result = U @ expm(deterministic) @ expm(1j * (Idx + Idy + Idz + Ir + Ip))

    return result*1j



########  Y - Gate  #########

def Y_gate_depolarization_noise_x(theta, ed):
    # Variances and covariances for depolarization Itô processes (X axis)
    Var_X = (2 * theta + np.sin(2 * theta)) / (4 * theta)
    Var_Y = 0
    Var_Z = (2 * theta - np.sin(2 * theta)) / (4 * theta)
    Cov_X_Y = 0
    Cov_X_Z = (np.sin(theta) ** 2) / (2 * theta)
    Cov_Y_Z = 0

    mean = np.array([0, 0, 0])
    cov = np.array([
        [Var_X,  Cov_X_Y,  Cov_X_Z],
        [Cov_X_Y, Var_Y,   Cov_Y_Z],
        [Cov_X_Z, Cov_Y_Z,  Var_Z]
    ])

    # Sample one realization: [dX, dY, dZ]
    sample_X = np.random.multivariate_normal(mean, cov,1)
    dX, dY, dZ = sample_X[0]

    # Construct Hermitian matrix: Id = dX·σ_X + dY·σ_Y + dZ·σ_Z
    Idx = ed * array([
        [dZ, dX - 1j * dY],
        [dX + 1j * dY, -dZ]
    ])
    return Idx


def Y_gate_depolarization_noise_y(theta, ed):
    """Simulates Hermitian noise matrix for depolarization along the Y axis."""
    # Explicit variances and covariances
    Var_X = 0
    Var_Y = 1
    Var_Z = 0

    Cov_X_Y = 0
    Cov_X_Z = 0
    Cov_Y_Z = 0

    mean = np.array([0, 0, 0])
    cov = np.array([
        [Var_X,  Cov_X_Y,  Cov_X_Z],
        [Cov_X_Y, Var_Y,   Cov_Y_Z],
        [Cov_X_Z, Cov_Y_Z,  Var_Z]
    ])

    # Sample one realization: [dX, dY, dZ]
    sample_Y = np.random.multivariate_normal(mean, cov,1)
    dX, dY, dZ = sample_Y[0]
    
    # Construct Hermitian matrix: Id = dX·σ_X + dY·σ_Y + dZ·σ_Z
    Idy = ed * array([
        [dZ, dX - 1j * dY],
        [dX + 1j * dY, -dZ]
    ])

    return Idy


def Y_gate_depolarization_noise_z(theta, ed):
    """Simulates Hermitian noise matrix for depolarization along the Y axis."""
    # Explicit variances and covariances
    Var_X = (2 * theta - np.sin(2 * theta)) / (4 * theta)
    Var_Y = 0
    Var_Z = (2 * theta + np.sin(2 * theta)) / (4 * theta)

    Cov_X_Y = 0
    Cov_X_Z = (np.sin(theta) ** 2) / (2 * theta)
    Cov_Y_Z = 0

    mean = np.array([0, 0, 0])
    cov = np.array([
        [Var_X,  Cov_X_Y,  Cov_X_Z],
        [Cov_X_Y, Var_Y,   Cov_Y_Z],
        [Cov_X_Z, Cov_Y_Z,  Var_Z]
    ])

    # Sample one realization: [dX, dY, dZ]
    sample_Z = np.random.multivariate_normal(mean, cov,1)
    dX, dY, dZ = sample_Z[0]
    # Construct Hermitian matrix: Id = dX·σ_X + dY·σ_Y + dZ·σ_Z
    Idz = ed * array([
        [dZ, dX - 1j * dY],
        [dX + 1j * dY, -dZ]
    ])

    return Idz


def Y_gate_deterministic_noise(e1, theta, phi):
    # 3 continued) Deterministic relaxation contribution
    det1 = (theta - np.sin(theta)) / (2 * theta)
    det2 = (np.cos(theta)-1) / (2 * theta)
    det3 = (theta + np.sin(theta)) / (2 * theta)

    deterministic = -e1**2 / 2 * array([
        [det1, 1j/2 * exp(-1j * phi) * det2],
        [-1j/2 * exp(1j * phi) * det2, det3]
    ])
    return deterministic


def Y_gate_construction(para_p, para_T1, para_T2, para_tg, additional_noise_bool = False):
    theta = np.pi  # Arbitrary phase, can be set to pi/2
    phi = np.pi / 2  # Y gate is a rotation of pi/2 around the Y axis
    psi = np.pi / 2  # Arbitrary phase, can be set to pi/2
    ed, e1, e2, ep = parameters(para_p, para_T1, para_T2, para_tg)

    U = make_U(theta,phi,psi)

    Idx = Y_gate_depolarization_noise_x(theta,ed)
    Idy = Y_gate_depolarization_noise_y(theta,ed)
    Idz = Y_gate_depolarization_noise_z(theta,ed)
    
    if not additional_noise_bool:
        result = U @ expm(1j * (Idx + Idy + Idz))
        
    else:
        Ir = e1*(Idx - 1j*Idy)/(ed*2)
        Ip = ep*Idz/ed
        deterministic = Y_gate_deterministic_noise(theta, phi, e1)
        result = U @ expm(deterministic) @ expm(1j * (Idx + Idy + Idz + Ir + Ip))

    return result



#########  Z - Gate  #########

def Z_gate_depolarization_noise_x(theta, ed):
    # Variances and covariances for depolarization Itô processes (X axis)
    Var_X = (2 * theta + np.sin(2 * theta)) / (4 * theta)
    Var_Y = (2 * theta - np.sin(2 * theta)) / (4 * theta)
    Var_Z = 0
    Cov_X_Y = (-np.sin(theta) ** 2) / (2 * theta)
    Cov_X_Z = 0
    Cov_Y_Z = 0

    mean = np.array([0, 0, 0])
    cov = np.array([
        [Var_X,  Cov_X_Y,  Cov_X_Z],
        [Cov_X_Y, Var_Y,   Cov_Y_Z],
        [Cov_X_Z, Cov_Y_Z,  Var_Z]
    ])

    # Sample one realization: [dX, dY, dZ]
    sample_X = np.random.multivariate_normal(mean, cov,1)
    dX, dY, dZ = sample_X[0]

    # Construct Hermitian matrix: Id = dX·σ_X + dY·σ_Y + dZ·σ_Z
    Idx = ed * array([
        [dZ, dX - 1j * dY],
        [dX + 1j * dY, -dZ]
    ])
    return Idx


def Z_gate_depolarization_noise_y(theta, ed):
    """Simulates Hermitian noise matrix for depolarization along the Y axis."""
    # Explicit variances and covariances
    Var_X = (2 * theta - np.sin(2 * theta)) / (4 * theta)
    Var_Y = (2 * theta + np.sin(2 * theta)) / (4 * theta)
    Var_Z = 0
    Cov_X_Y = (np.sin(theta) ** 2) / (2 * theta)
    Cov_X_Z = 0
    Cov_Y_Z = 0

    mean = np.array([0, 0, 0])
    cov = np.array([
        [Var_X,  Cov_X_Y,  Cov_X_Z],
        [Cov_X_Y, Var_Y,   Cov_Y_Z],
        [Cov_X_Z, Cov_Y_Z,  Var_Z]
    ])

    # Sample one realization: [dX, dY, dZ]
    sample_Y = np.random.multivariate_normal(mean, cov,1)
    dX, dY, dZ = sample_Y[0]
    
    # Construct Hermitian matrix: Id = dX·σ_X + dY·σ_Y + dZ·σ_Z
    Idy = ed * array([
        [dZ, dX - 1j * dY],
        [dX + 1j * dY, -dZ]
    ])

    return Idy


def Z_gate_depolarization_noise_z(theta, ed):
    """Simulates Hermitian noise matrix for depolarization along the Y axis."""
    # Explicit variances and covariances
    Var_X = 0
    Var_Y = 0
    Var_Z = 1

    Cov_X_Y = 0
    Cov_X_Z = 0
    Cov_Y_Z = 0

    mean = np.array([0, 0, 0])
    cov = np.array([
        [Var_X,  Cov_X_Y,  Cov_X_Z],
        [Cov_X_Y, Var_Y,   Cov_Y_Z],
        [Cov_X_Z, Cov_Y_Z,  Var_Z]
    ])

    # Sample one realization: [dX, dY, dZ]
    sample_Z = np.random.multivariate_normal(mean, cov,1)
    dX, dY, dZ = sample_Z[0]
    # Construct Hermitian matrix: Id = dX·σ_X + dY·σ_Y + dZ·σ_Z
    Idz = ed * array([
        [dZ, dX - 1j * dY],
        [dX + 1j * dY, -dZ]
    ])

    return Idz


def Z_gate_deterministic_noise(e1):
    deterministic = -e1**2 / 2 * array([
        [0,0],
        [0, 1]
    ])
    return deterministic


def Z_gate_construction(para_p, para_T1, para_T2, para_tg, additional_noise_bool = False):
    theta = np.pi  # Arbitrary phase, can be set to pi/2
    phi = np.pi / 2  # Z gate is a rotation of pi/2 around the Z axis
    psi = 0
    ed, e1, e2, ep = parameters(para_p, para_T1, para_T2, para_tg)

    U = make_U(theta,phi,psi)

    Idx = Z_gate_depolarization_noise_x(theta,ed)
    Idy = Z_gate_depolarization_noise_y(theta,ed)
    Idz = Z_gate_depolarization_noise_z(theta,ed)

    if not additional_noise_bool:
        result = U @ expm(1j * (Idx + Idy + Idz))
        
    else:
        Ir = e1*(Idx - 1j*Idy)/(ed*2)
        Ip = ep*Idz/ed
        deterministic = Z_gate_deterministic_noise(e1)
        result = U @ expm(deterministic) @ expm(1j * (Idx + Idy + Idz + Ir + Ip))

    return result*1j


##### H - Gate #####
def H_gate_construction(para_p, para_T1, para_T2, para_tg, additional_noise_bool = False):
    X_noisy_gate = X_gate_construction(para_p, para_T1, para_T2, para_tg, additional_noise_bool)
    Z_noisy_gate = Z_gate_construction(para_p, para_T1, para_T2, para_tg, additional_noise_bool)
    
    H_noisy_gate = 1/(2**0.5)*(X_noisy_gate + Z_noisy_gate)  # 4x4 Hadamard gate with noise
    return H_noisy_gate