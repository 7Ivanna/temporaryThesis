import numpy as np
import matplotlib.pyplot as plt
from quotonic.fock import basis
from quotonic.aa import multiPhotonUnitary

def gaussian(t, tj, tau, sigma_t):
    normalization_factor = 1 / (np.sqrt(sigma_t) * np.sqrt(np.sqrt( np.pi)))
    return normalization_factor * np.exp(-((t - tj - tau) ** 2) / (2 * sigma_t**2))

fb = basis(2, 2)
U = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
multiguy = multiPhotonUnitary(2, U)

tj = 3.0
t = np.linspace(-10, 10, 200)
t1, t2 = np.meshgrid(t, t)

alpha1_t1 = gaussian(t1, -tj / 2.0, 0.0, 1.0)
alpha1_t2 = gaussian(t2, -tj / 2.0, 0.0, 1.0)
alpha2_t1 = gaussian(t1, tj / 2.0, 0.0, 1.0)
alpha2_t2 = gaussian(t2, tj / 2.0, 0.0, 1.0)
# wf = alpha1_t1 * alpha2_t2 + alpha1_t2 * alpha2_t1
# wf /= np.sqrt(np.trapz(np.trapz(np.abs(wf) ** 2, t), t))
wf = alpha1_t1 * alpha2_t2

psi_in = np.zeros((3, 200, 200), dtype=complex)
psi_in[0] = wf / np.sqrt(2.0)
psi_in[2] = -wf / np.sqrt(2.0)
psi_out = np.tensordot(multiguy, psi_in, axes=1)
print(np.trapz(np.trapz(np.abs(psi_out) ** 2, t), t))

V = 0.5 - 0.5 * np.abs(np.trapz(gaussian(t, -tj / 2.0, 0.0, 1.0) * gaussian(t, tj / 2.0, 0.0, 1.0), t)) ** 2
print(V)
V2 = 0.25 * np.trapz(np.trapz(np.abs(wf - np.flip(wf)) ** 2, t), t)
print(V2)
