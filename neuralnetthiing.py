from quotonic.clements import Mesh
import quotonic.qpnn as qp
import quotonic.kerr as kerr
import quotonic.fock as fock
import numpy as np
import quotonic.aa as aa
from quotonic.fock import basis, getDim
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import quotonic.training_sets as ts
from quotonic.misc import genHaarUnitary
# N=6
# n=2 #cause girl we got 2 photons
# m=6 #she be a 6 x 6 matrix right
# D=np.array( #the D array
#             [
#                 [1, 0,0,0,0,0],
#                 [0,1,0,0,0,0],
#                 [0,0,1,0,0,0],
#                 [0,0,0,1,0,0],
#                 [0,0,0,0,1,0],
#                 [0,0,0,0,0,1]
#             ],dtype=complex)
# mesh = Mesh(6)
# U = np.array([[-1, np.sqrt(2), 0, 0, 0, 0],
#         [np.sqrt(2), 1, 0, 0, 0, 0],
#         [0, 0, -1, 1, 1, 0],
#         [0, 0, 1, 1, 0, 1],
#         [0, 0, 1, 0, 1, -1],
#         [0, 0, 0, 1, -1, -1]], dtype=complex) / np.sqrt(3)
# print(mesh.decode(U))

# print("the phases are", mesh.phases)

# new=np.dot(D,U)

# multiguy=aa.multiPhotonUnitary(2, new)# I think this is fine
# # multiguy = np.square(np.abs(multiguy))
# myfockbasis=basis(n,m)
# N=len(myfockbasis)
# ind = [myfockbasis.index([0, 1, 0, 1, 0, 0]),
#         myfockbasis.index([0, 1, 0, 0, 1, 0]),
#         myfockbasis.index([0, 0, 1, 1, 0, 0]),
#         myfockbasis.index([0, 0, 1, 0, 1, 0])]  #these seem... alright?

#generating harr

train=ts.CNOT()
neural=qp.QPNN(2,4,2,trainingSet=train)

allPhases = np.zeros(neural.pspl * 2)

harrmesh = Mesh(4)
random_matrix = genHaarUnitary(4)
decoded = harrmesh.decode(random_matrix)
allPhases[0 : neural.pspl] = harrmesh.phases

random_matrix = genHaarUnitary(4)
decoded = harrmesh.decode(random_matrix)
allPhases[neural.pspl ::] = harrmesh.phases

print("meshes", allPhases)
print("ok")

#kerr_unit = kerr.buildKerrUnitary(2,6,np.pi)
#print(kerr_unit)
#u_k_array = np.dot(kerr_unit,multiguy)

setting =  neural.set_phases(allPhases)
func = neural.sysFunc()
#print(neural)

# Gaussian function
def gaussian(t, tj, tau, sigma_t):
            normalization_factor = 1 / (sigma_t * np.sqrt(np.sqrt( np.pi)))
            return normalization_factor * np.exp(-((t - tj - tau) ** 2) / (2 * sigma_t**2))

sigma_t = 1 / (2 * np.sqrt(2 * np.log(2))) # converted sigma

t = np.linspace(-10, 10, 200)
t1, t2 = np.meshgrid(t, t)
tj = 0.0
alpha1_t1 = gaussian(t1, -tj / 2.0, 0.0, 1.0)
alpha1_t2 = gaussian(t1, -tj / 2.0, 0.0, 1.0)
alpha2_t1 = gaussian(t1, tj / 2.0, 0.0, 1.0)
alpha2_t2 = gaussian(t2, tj / 2.0, 0.0, 1.0)
wf = 0.5 * (alpha1_t1 * alpha2_t2 + alpha1_t2 * alpha2_t1)

psi0 = np.zeros((10, 200, 200), dtype = complex)

# if you want |1010> as your input state, then you need to find that and insert wf there
ind = basis(2, 4).index([1, 0, 1, 0])
psi0[ind, :, :] = wf

psi_out = np.tensordot(func, psi0, axes=1 )
print(psi_out)
t = np.linspace(-10, 10, 200)
tj =0
tau = np.linspace(-10, 10, 200)

