import numpy as np
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
from itertools import product


def basis (num_modes, num_photons):
    photons_array=np.arange(1,num_photons+1,1)
    states=(list(product(photons_array, repeat = num_modes)))
    # print(states)
    return states




# Define U from Jacobs code (its in his basis)

small_U = np.array([[-1, np.sqrt(2), 0, 0, 0, 0],  # U for 1 photon
              [np.sqrt(2), 1, 0, 0, 0, 0],
              [0, 0, -1, 1, 1, 0],
              [0, 0, 1, 1, 0, 1],
              [0, 0, 1, 0, 1, -1],
              [0, 0, 0, 1, -1, -1]], dtype=complex) / np.sqrt(3)

numPhotons = 2 

numModes = 6

small_U = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0) 
numPhotons = 2
numModes = 2

U = small_U

for i in range( numPhotons - 1 ):
    U = np.kron( U , small_U)

# Gaussian function
def gaussian(t, tj, tau, sigma_t):
            normalization_factor = 1 / (sigma_t * np.sqrt(np.sqrt( np.pi)))
            return normalization_factor * np.exp(-((t - tj - tau) ** 2) / (2 * sigma_t**2))

sigma_t = 1 / (2 * np.sqrt(2 * np.log(2))) # converted sigma

t = np.linspace(-10, 10, 200)
t1, t2 = np.meshgrid(t, t)
tj = 3.0
alpha1_t1 = gaussian(t1, -tj / 2.0, 0.0, 1.0)
alpha1_t2 = gaussian(t2, -tj / 2.0, 0.0, 1.0)
alpha2_t1 = gaussian(t1, tj / 2.0, 0.0, 1.0)
alpha2_t2 = gaussian(t2, tj / 2.0, 0.0, 1.0)
#wf = 0.5 * (alpha1_t1 * alpha2_t2 + alpha1_t2 * alpha2_t1)

wf12 = alpha1_t1 * alpha2_t2 / np.sqrt(2.0)
wf21 = alpha1_t2 * alpha2_t1 / np.sqrt(2.0)
print(wf12)
# wf12 = (alpha1_t1 * alpha2_t2) / np.sqrt(2.0)
# wf21 = -(alpha1_t2 * alpha2_t1) / np.sqrt(2.0)
# wf12 /= np.sqrt(np.trapz(np.trapz(np.abs(wf12) ** 2, t), t))

# print(wf12)
# wf12 /= np.sqrt(2.0)
psi0 = np.zeros((36, 200, 200), dtype = complex)
psi2 = np.zeros((36, 200, 200), dtype = complex)
psi3 = np.zeros((36, 200, 200), dtype = complex)
psi4 = np.zeros((36, 200, 200), dtype = complex)
print(psi0.shape)
# if you want |1010> as your input state, then you need to find that and insert wf there
ind = basis(2, 6).index((2, 4))  # Ensure tuple format
psi0[ind, :, :] = wf12
ind2 = basis(2, 6).index((2, 5))
psi2[ind2, :, :] = wf12
ind3 = basis(2, 6).index((3,4))
psi3[ind3, :, :] = wf12
ind4 = basis(2, 6).index((3,5))
psi4[ind4, :, :] = wf12
ind = basis(2, 6).index((4, 2))  # Ensure tuple format
psi0[ind, :, :] = wf21
ind2 = basis(2, 6).index((5, 2))
psi2[ind2, :, :] = wf21
ind3 = basis(2, 6).index((4,3))
psi3[ind3, :, :] = wf21
ind4 = basis(2, 6).index((5,3))
psi4[ind4, :, :] = wf21
print(np.shape(psi0))
psi_out = np.tensordot(U, psi0, axes=1 )
psi_out2 = np.tensordot(U, psi2, axes=1 )
psi_out3 = np.tensordot(U, psi3, axes=1 )
psi_out4 = np.tensordot(U, psi4, axes=1 )

#print(psi_out)
t = np.linspace(-10, 10, 200)
tj =np.arange(0,3,200)
tau = np.linspace(0, 10, 200)
print(len(tj))
# this calculates the visibility ; same as only half of the hom measurment ! 
def integral(t, tj,tau):
    total = np.zeros(len(tau))
    for i in range(len(tau)):
        t_val = t
        # Compute the overlap integral
        integral_V_part1 = np.trapz(np.conj(gaussian(t_val, 0, 0, 1)) * gaussian(t_val, tj, tau[i], 1),t_val)
        integral_V_part2= np.trapz(np.conj(gaussian(t_val, tj, tau[i], 1))* gaussian(t_val, 0, 0, 1),t_val )

        total[i] = np.square(integral_V_part1*integral_V_part2) 
    
        
    return tau,total
V_array=np.zeros(len(tj))
print("ok")
for i in range(len(tj)):
    x,y=integral(t, tj,tau)
    print(x.size, y.size)
    y_max = np.max(y)
    y_min = np.min(y)
    V = ( y_max - y_min )/y_max
    print(V)
    V_array[i] =V
   
plt.plot(x,y)
plt.legend()
plt.title('Overlap')
plt.xlabel(' Mean Time Delay [ns]')
plt.ylabel('Overlap')
plt.grid(True)
plt.show()

# calculating the target wavefunction

def hinton(matrix, x_labels=None, y_labels=None, max_weight=None, ax=None): #hinton timeeeeeeeeeee
    ax = ax if ax is not None else plt.gca()
    if not max_weight:
        max_weight = np.max(np.abs(matrix))

    # Set the background color of the axes
    ax.set_facecolor('white')
    ax.set_title('CNOT Gate Results for a 2 Photon System with Unconditional Fidelity', fontsize=13, wrap='true' ,pad=10)
    # Define the colormap for Hinton diagram
    cmap = 'RdPu'

    im = ax.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=0, vmax=max_weight)

    ax.autoscale_view()
    #ax.invert_yaxis()
    ax.set_ylabel("Output State")
    ax.set_xlabel("Input State")
    if x_labels is None:
        x_labels = ['|00⟩','|01⟩', '|10⟩', '|11⟩']
       
    if y_labels is None:
         y_labels = ['⟨00|','⟨01|', '⟨10|', '⟨11|']
    ax.set_xticks(np.arange(len(x_labels)), )
    ax.set_xticklabels(x_labels)

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.5, color='black')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fidelity')

def probabilityCalc(psi0_target,psi2_target,psi3_target,psi4_target,psi_out,psi_out2,psi_out3,psi_out4,t):
    e11=np.tensordot(psi0_target,psi_out, axes=1)
    e12=np.tensordot(psi0_target,psi_out2, axes=1)
    e13=np.tensordot(psi0_target,psi_out3, axes=1)
    e14=np.tensordot(psi0_target,psi_out4, axes=1)
    
    element11=np.trapz(np.trapz(np.abs(e11) ** 2, t), t) #how to do the 2d integral\
    element12=np.trapz(np.trapz(np.abs(e12) ** 2, t), t) #how to do the 2d integral
    element13=np.trapz(np.trapz(np.abs(e13) ** 2, t), t) #how to do the 2d integral\
    element14=np.trapz(np.trapz(np.abs(e14) ** 2, t), t) #how to do the 2d integral

    e21=np.tensordot(psi2_target,psi_out, axes=1)
    e22=np.tensordot(psi2_target,psi_out2, axes=1)
    e23=np.tensordot(psi2_target,psi_out3, axes=1)
    e24=np.tensordot(psi2_target,psi_out4, axes=1)
    element21=np.trapz(np.trapz(np.abs(e21) ** 2, t), t) #how to do the 2d integral\
    element22=np.trapz(np.trapz(np.abs(e22) ** 2, t), t) #how to do the 2d integral
    element23=np.trapz(np.trapz(np.abs(e23) ** 2, t), t) #how to do the 2d integral\
    element24=np.trapz(np.trapz(np.abs(e24) ** 2, t), t) #how to do the 2d integral

    e31=np.tensordot(psi3_target,psi_out, axes=1)
    e32=np.tensordot(psi3_target,psi_out2, axes=1)
    e33=np.tensordot(psi3_target,psi_out3, axes=1)
    e34=np.tensordot(psi3_target,psi_out4, axes=1)
    element31=np.trapz(np.trapz(np.abs(e31) ** 2, t), t) #how to do the 2d integral\
    element32=np.trapz(np.trapz(np.abs(e32) ** 2, t), t) #how to do the 2d integral
    element33=np.trapz(np.trapz(np.abs(e33) ** 2, t), t) #how to do the 2d integral\
    element34=np.trapz(np.trapz(np.abs(e34) ** 2, t), t) #how to do the 2d integral

    e41=np.tensordot(psi4_target,psi_out, axes=1)
    e42=np.tensordot(psi4_target,psi_out2, axes=1)
    e43=np.tensordot(psi4_target,psi_out3, axes=1)
    e44=np.tensordot(psi4_target,psi_out4, axes=1)
    element41=np.trapz(np.trapz(np.abs(e41) ** 2, t), t) #how to do the 2d integral\
    element42=np.trapz(np.trapz(np.abs(e42) ** 2, t), t) #how to do the 2d integral
    element43=np.trapz(np.trapz(np.abs(e43) ** 2, t), t) #how to do the 2d integral\
    element44=np.trapz(np.trapz(np.abs(e44) ** 2, t), t) #how to do the 2d integral

    full4by4= np.array([[element11, element21, element31, element41], 
               [element12,element22,element32,element42], 
               [element13,element23,element33,element43],
               [element14,element24,element34,element44]])

    avg_probability =  [(sum(x))**(-1) for x in zip(*full4by4)]
    print(full4by4)
    
    plt.figure()
    y_labels = ['⟨00|','⟨01|', '⟨10|', '⟨11|']
    x_labels = ['|00⟩','|01⟩', '|10⟩', '|11⟩']
    hinton(np.real(full4by4), x_labels, y_labels)
    plt.xticks(np.arange(len(x_labels)), [str(state) for state in x_labels])
    plt.yticks(np.arange(len(y_labels)), [str(state) for state in y_labels])
    plt.show()

    return avg_probability,psi0_target,psi2_target,psi3_target,psi4_target,psi_out,psi_out2,psi_out3,psi_out4,t

def ConditionalProbReal(avg_probability,psi0_target,psi2_target,psi3_target,psi4_target,psi_out,psi_out2,psi_out3,psi_out4,t):
    e11=np.tensordot(psi0_target,psi_out, axes=1)
    e12=np.tensordot(psi0_target,psi_out2, axes=1)
    e13=np.tensordot(psi0_target,psi_out3, axes=1)
    e14=np.tensordot(psi0_target,psi_out4, axes=1)
    element11=np.trapz(np.trapz(np.abs(e11) ** 2, t), t) 
    element12=np.trapz(np.trapz(np.abs(e12) ** 2, t), t) 
    element13=np.trapz(np.trapz(np.abs(e13) ** 2, t), t) 
    element14=np.trapz(np.trapz(np.abs(e14) ** 2, t), t) 

    e21=np.tensordot(psi2_target,psi_out, axes=1)
    e22=np.tensordot(psi2_target,psi_out2, axes=1)
    e23=np.tensordot(psi2_target,psi_out3, axes=1)
    e24=np.tensordot(psi2_target,psi_out4, axes=1)
    element21=np.trapz(np.trapz(np.abs(e21) ** 2, t), t) 
    element22=np.trapz(np.trapz(np.abs(e22) ** 2, t), t) 
    element23=np.trapz(np.trapz(np.abs(e23) ** 2, t), t) 
    element24=np.trapz(np.trapz(np.abs(e24) ** 2, t), t) 

    e31=np.tensordot(psi3_target,psi_out, axes=1)
    e32=np.tensordot(psi3_target,psi_out2, axes=1)
    e33=np.tensordot(psi3_target,psi_out3, axes=1)
    e34=np.tensordot(psi3_target,psi_out4, axes=1)
    element31=np.trapz(np.trapz(np.abs(e31) ** 2, t), t) 
    element32=np.trapz(np.trapz(np.abs(e32) ** 2, t), t) 
    element33=np.trapz(np.trapz(np.abs(e33) ** 2, t), t)
    element34=np.trapz(np.trapz(np.abs(e34) ** 2, t), t)

    e41=np.tensordot(psi4_target,psi_out, axes=1)
    e42=np.tensordot(psi4_target,psi_out2, axes=1)
    e43=np.tensordot(psi4_target,psi_out3, axes=1)
    e44=np.tensordot(psi4_target,psi_out4, axes=1)
    element41=np.trapz(np.trapz(np.abs(e41) ** 2, t), t)
    element42=np.trapz(np.trapz(np.abs(e42) ** 2, t), t)
    element43=np.trapz(np.trapz(np.abs(e43) ** 2, t), t)
    element44=np.trapz(np.trapz(np.abs(e44) ** 2, t), t)
    full4by4= np.array([[element11, element21, element31, element41], 
               [element12,element22,element32,element42], 
               [element13,element23,element33,element43],
               [element14,element24,element34,element44]])
    
    final = np.array([i*avg_probability for i in full4by4])
    
    print(final)

    return


print(np.trapz(np.trapz(np.abs(psi_out[basis(2,6).index((2,4))]) ** 2, t), t) + np.trapz(np.trapz(np.abs(psi_out[basis(2,6).index((4,2))]) ** 2, t), t))
print(np.trapz(np.trapz(np.abs(psi_out2[basis(2,6).index((2,5))]) ** 2, t), t) + np.trapz(np.trapz(np.abs(psi_out2[basis(2,6).index((5,2))]) ** 2, t), t))
print(np.trapz(np.trapz(np.abs(psi_out3[basis(2,6).index((3,5))]) ** 2, t), t) + np.trapz(np.trapz(np.abs(psi_out3[basis(2,6).index((5,3))]) ** 2, t), t))
print(np.trapz(np.trapz(np.abs(psi_out4[basis(2,6).index((3,4))]) ** 2, t), t) + np.trapz(np.trapz(np.abs(psi_out4[basis(2,6).index((4,3))]) ** 2, t), t))

psi0_target = np.zeros(36, dtype=complex)
psi2_target = np.zeros(36, dtype=complex)
psi3_target = np.zeros(36, dtype=complex)
psi4_target = np.zeros(36, dtype=complex)

ind_psi_targ = basis(2, 6).index((2,4))
psi0_target[ind_psi_targ] = 1
ind_psi_targ2 = basis(2, 6).index((2,5))
psi2_target[ind_psi_targ2] = 1
ind_psi_targ4 = basis(2, 6).index((3,5))

ind_psi_targ3 = basis(2, 6).index((3,4))
psi4_target[ind_psi_targ3] = 1 #ask jacob about this.... casue im sus as fukkk
psi3_target[ind_psi_targ4] = 1


avg_probability,psi0_target,psi2_target,psi3_target,psi4_target,psi_out,psi_out2,psi_out3,psi_out4,t=probabilityCalc(psi0_target,psi2_target,psi3_target,psi4_target,psi_out,psi_out2,psi_out3,psi_out4,t) #sending this off to calculate the probabilites

ConditionalProbReal(avg_probability,psi0_target,psi2_target,psi3_target,psi4_target,psi_out,psi_out2,psi_out3,psi_out4,t)

