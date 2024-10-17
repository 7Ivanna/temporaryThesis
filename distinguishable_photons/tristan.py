
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd

def alpha(t, tj,tau):
        return ((2 * np.pi) ** -0.25) * np.exp(-0.25 * ((t - tj - tau) ** 2), dtype=complex) # gaussian, should be .5 instead of .25? 

def gen_basis(numPhotons, numModes):
        return list(product(list(range(1, numModes + 1, 1)), repeat=numPhotons))

def compute_wf_norm(t, psi):
        return np.trapz(np.trapz(np.abs(psi) ** 2, t), t)


def compute_visibility(tj):
    t = np.linspace(-10, 10, 200)
    total = 0.5 - 0.5 * np.abs(np.trapz(np.conj(alpha(t, tj / 2, 0)) * alpha(t, -tj / 2, 0), t)) ** 2
    return total

def compute_conditional_fidelity(tj):
    t = np.linspace(-10, 10, 200)
    t1, t2 = np.meshgrid(t, t)
    wf12 = alpha(t1, -tj / 2.0,0) * alpha(t2, tj / 2.0,0) / np.sqrt(2.0)
    wf21 = alpha(t1, tj / 2.0,0) * alpha(t2, -tj / 2.0,0) / np.sqrt(2.0)

    numPhotons = 2
    numModes = 6
    basis = gen_basis(numPhotons, numModes)
    dim = len(basis)
    comp_basis_inds = [[basis.index((2,4)), basis.index((4,2))], [basis.index((2,5)), basis.index((5,2))], [basis.index((3,4)), basis.index((4,3))], [basis.index((3,5)), basis.index((5,3))]]

    U = np.array([[-1.0, np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0], [np.sqrt(2.0), 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0, -1.0, -1.0]], dtype=complex) / np.sqrt(3.0)
    PhiU = np.kron(U, U)

    targets = [(0, 1), (2, 3), (6, 7), (4, 5)]

    conditional_fidelity = 0.0
    for i, cb_ind in enumerate(comp_basis_inds):
        psi = np.zeros((dim, Nt, Nt), dtype=complex)
        psi[cb_ind[0]] = wf12
        psi[cb_ind[1]] = wf21
        psi_out = np.tensordot(PhiU, psi, axes=1)

        norm_sum = 0
        psi_out_logical = np.zeros((2*len(comp_basis_inds), Nt, Nt), dtype=complex)
        for j in range(0, 2*len(comp_basis_inds), 2):
            psi_out_logical[j] = psi_out[comp_basis_inds[j // 2][0]]
            psi_out_logical[j+1] = psi_out[comp_basis_inds[j // 2][1]]
            norm_sum += compute_wf_norm(t, psi_out_logical[j]) + compute_wf_norm(t, psi_out_logical[j+1])
        psi_out_logical /= np.sqrt(norm_sum)

        conditional_fidelity += compute_wf_norm(t, psi_out_logical[targets[i][0]]) + compute_wf_norm(t, psi_out_logical[targets[i][1]])
    conditional_fidelity /= len(comp_basis_inds)   
    return conditional_fidelity

Nt = 200
tlim = 10

tjs = np.linspace(0,5.5,100)
V = np.zeros(100)
F = np.zeros(100)
for i, tj in enumerate(tjs):
    V[i] = compute_visibility(tj)
    #print(V)
    F[i] = compute_conditional_fidelity(tj)


data_dict = {
    'Visibility': V,
    'Fidelity': F
}

df = pd.DataFrame(data_dict)
csv_file_path = 'data.csv'
df.to_csv(csv_file_path, index=False)

plt.plot(tjs, V)
plt.plot(tjs, F)
plt.show()

plt.plot(V, F)

plt.ylabel('Fidelity Average')
plt.xlabel('Visibility')
plt.show()
