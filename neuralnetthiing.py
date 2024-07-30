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

setting_phases =  neural.set_phases(allPhases)
func_S = neural.sysFunc()

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
ind2 = basis(2, 4).index([1, 0, 0, 1])
psi0[ind2, :, :] = wf
ind3 = basis(2, 4).index([0, 1, 1, 0])
psi0[ind3, :, :] = wf
ind4 = basis(2, 4).index([0, 1, 0, 1])
psi0[ind3, :, :] = wf

psi_out = np.tensordot(func_S, psi0, axes=1 )
print(psi_out)
t = np.linspace(-10, 10, 200)
tj =0
tau = np.linspace(-10, 10, 200)

psi0_act = np.zeros((10, 200, 200), dtype = complex)

ind_psi_act = basis(2, 4).index([1, 0, 1, 0])
psi0_act[ind_psi_act, :, :] = 1.0
ind_psi_act2 = basis(2, 4).index([1, 0, 0, 1])
psi0_act[ind_psi_act2, :, :] = 1.0
ind_psi_act3 = basis(2, 4).index([0, 1, 1, 0])
psi0_act[ind_psi_act3, :, :] = 1.0
ind_psi_act4 = basis(2, 4).index([0, 1, 0, 1])
psi0_act[ind_psi_act4, :, :] = 1.0

print("dot product")
psi_out_act_psi_out = np.dot(psi0_act, psi_out )
#print(psi_out_act_psi_out)

psi_out_act_psi_out1= basis(2, 4).index([1, 0, 1, 0])
psi_out_act_psi_out2= basis(2, 4).index([1, 0, 0, 1])

print(len(psi_out_act_psi_out))


# full4by4= 1/21*np.square(np.abs(np.array([[element11, element21, element31, element41], 
#                [ element12,element22,element32,element42], 
#                [element13,element23,element33,element43],
#                [element14,element24,element34,element44]])))




