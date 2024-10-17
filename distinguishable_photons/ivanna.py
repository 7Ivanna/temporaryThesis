import numpy as np
from itertools import product

def alpha(t, t0):
    return ((2 * np.pi) ** -0.25) * np.exp(-0.25 * ((t - t0) ** 2), dtype=complex) # gaussian, should be .5 instead of .25? 

def gen_basis(numPhotons, numModes):
    return list(product(list(range(1, numModes + 1, 1)), repeat=numPhotons))

def compute_wf_norm(t, psi):
    return np.trapz(np.trapz(np.abs(psi) ** 2, t), t)
total_sum = 0.0
non_zero_count = 0

tj = 0
Nt = 200
tlim = 10.0
t = np.linspace(-tlim, tlim, Nt)
t1, t2 = np.meshgrid(t, t)
wf12 = alpha(t1, -tj / 2.0) * alpha(t2, tj / 2.0) / np.sqrt(2.0)
wf21 = alpha(t1, tj / 2.0) * alpha(t2, -tj / 2.0) / np.sqrt(2.0)

""" 50:50 BS """
print("50:50 BS")
numPhotons = 2
numModes = 2
basis = gen_basis(numPhotons, numModes)
dim = len(basis)

U = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
PhiU = np.kron(U, U)

psi = np.zeros((len(basis), Nt, Nt), dtype=complex)
psi[basis.index((1,2))] = wf12
psi[basis.index((2,1))] = wf21
psi_out = np.tensordot(PhiU, psi, axes=1)
result = ""
for i in range(dim):
    result += f"|{basis[i][0]:d}{basis[i][1]:d}> -> {compute_wf_norm(t, psi_out[i]):.4f}\t\t"
    # print(f"|{basis[i][0]:d}{basis[i][1]:d}> -> {compute_wf_norm(t, psi_out[i]):.4f}")
print(result + "\n")

""" CNOT """
print("CNOT")
numPhotons = 2
numModes = 6
basis = gen_basis(numPhotons, numModes)
dim = len(basis)
comp_basis_inds = [[basis.index((2,4)), basis.index((4,2))], [basis.index((2,5)), basis.index((5,2))], [basis.index((3,4)), basis.index((4,3))], [basis.index((3,5)), basis.index((5,3))]]

U = np.array([[-1.0, np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0], [np.sqrt(2.0), 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0, -1.0, -1.0]], dtype=complex) / np.sqrt(3.0)
PhiU = np.kron(U, U)

for i, cb_ind in enumerate(comp_basis_inds):
    psi = np.zeros((dim, Nt, Nt), dtype=complex)
    psi[cb_ind[0]] = wf12
    psi[cb_ind[1]] = wf21
    psi_out = np.tensordot(PhiU, psi, axes=1)
    result = ""
    for j, cb_ind in enumerate(comp_basis_inds):
        norm_sum = compute_wf_norm(t, psi_out[cb_ind[0]]) + compute_wf_norm(t, psi_out[cb_ind[1]])
        result += f"{norm_sum:.4f}\t\t"
        
        # Only sum non-zero values
        if norm_sum != 0.0000 :
            total_sum += norm_sum
            non_zero_count += 1
    
    print(result)


    average = total_sum / 4
    print(non_zero_count)
    print(f"\nAverage of non-zero results: {average:.4f}")
