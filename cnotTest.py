from quotonic.clements import Mesh
import numpy as np
import quotonic.aa as aa
from quotonic.fock import basis, getDim
import matplotlib.pyplot as plt
from matplotlib import colors


def hinton(matrix, x_labels=None, y_labels=None, max_weight=None, ax=None): #hinton timeeeeeeeeeee
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = np.max(np.abs(matrix))

    # Set the background color of the axes
    ax.set_facecolor('white')
    ax.set_title('CNOT Gate Results for a 2 Photon System with Conditional Fidelity', fontsize=13, wrap='true' ,pad=10)
    # Define the colormap for Hinton diagram
    cmap = 'RdPu'
    ax.autoscale_view()
    im = ax.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=0, vmax=max_weight)

    ax.autoscale_view()
    #ax.invert_yaxis()
    ax.set_ylabel("Output State")
    ax.set_xlabel("Input State")
    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)

    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.5, color='black')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fidelity')
    plt.tight_layout()  #



def decodeCNOT(): #cnot to decode
     mesh = Mesh(6)
     U = np.array([[-1, np.sqrt(2), 0, 0, 0, 0],
            [np.sqrt(2), 1, 0, 0, 0, 0],
            [0, 0, -1, 1, 1, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, -1],
            [0, 0, 0, 1, -1, -1]], dtype=complex) / np.sqrt(3)
     print(mesh.decode(U))
 
     print("the phases are", mesh.phases)
decodeCNOT()
U = np.array( #the U array bc im dumb 
        [
         [-1, np.sqrt(2),0,0,0,0],
            [np.sqrt(2)/np.sqrt(3),1/np.sqrt(3),0,0,0,0],
            [0,0,-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),0],
            [0,0,1/np.sqrt(3),1/np.sqrt(3),0,1/np.sqrt(3)],
            [0,0,1/np.sqrt(3),0,1/np.sqrt(3),-1/np.sqrt(3)],
            [0,0,0,1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)]
        ],dtype=complex)

def multiPhotonUnitary(): #ngl idk why this is here
    mesh = Mesh(6)
    numPhotons = 2
    U = np.array( # well this didnt work :)
        [
         [-1, np.sqrt(2),0,0,0,0],
            [np.sqrt(2)/np.sqrt(3),1/np.sqrt(3),0,0,0,0],
            [0,0,-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),0],
            [0,0,1/np.sqrt(3),1/np.sqrt(3),0,1/np.sqrt(3)],
            [0,0,1/np.sqrt(3),0,1/np.sqrt(3),-1/np.sqrt(3)],
            [0,0,0,1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)]
        ],
        dtype=complex,
    )
    mesh.decode(U)
    return 
#print(aa.multiPhotonUnitary(2, U))  #so i have to pass the attributes to it.... ?
def probabilityCalc(e1,e2,e3,e4,psi1,psi2,psi3,psi4):
#I hope the code gods forgive my sins
    element11=np.dot(psi1,e1)
    element12=np.dot(psi1,e2)
    element13=np.dot(psi1,e3)
    element14=np.dot(psi1,e4)

    element21=np.dot(psi2,e1)
    element22=np.dot(psi2,e2)
    element23=np.dot(psi2,e3)
    element24=np.dot(psi2,e4)

    element31=np.dot(psi3,e1)
    element32=np.dot(psi3,e2)
    element33=np.dot(psi3,e3)
    element34=np.dot(psi3,e4)

    element41=np.dot(psi4,e1)
    element42=np.dot(psi4,e2)
    element43=np.dot(psi4,e3)
    element44=np.dot(psi4,e4)
    '''element11=np.dot(psi1,e1*0.4446) #this is most likely wrong, dont really have the time to look it over rns
    element12=np.dot(psi1,e2*0.4446)
    element13=np.dot(psi1,e3*0.14813)
    element14=np.dot(psi1,e4*0.14813)

    element21=np.dot(psi2,e1*0.4446)
    element22=np.dot(psi2,e2*0.4446)
    element23=np.dot(psi2,e3*0.14813)
    element24=np.dot(psi2,e4*0.14813)

    element31=np.dot(psi3,e1*0.4446)
    element32=np.dot(psi3,e2*0.4446)
    element33=np.dot(psi3,e3*0.14813)
    element34=np.dot(psi3,e4*0.14813)

    element41=np.dot(psi4,e1*0.4446)
    element42=np.dot(psi4,e2*0.4446)
    element43=np.dot(psi4,e3*0.14813)
    element44=np.dot(psi4,e4*0.14813)'''
#again im sorry code gods , here is my go
    full4by4= [[element11**2, element21**2, element31**2, element41**2], 
               [ element12**2,element22**2,element32**2,element42**2], 
               [element13**2,element23**2,element33**2,element43**2],
               [element14**2,element24**2,element34**2,element44**2]]
    myfockbasis = [
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0]]
    plt.figure() #oh wow much plot 
    
    y_labels = ['⟨00|','⟨01|', '⟨10|', '⟨11|']
    x_labels = ['|00⟩','|01⟩', '|10⟩', '|11⟩']
    hinton(np.real(full4by4)*9, x_labels, y_labels)
    plt.xticks(np.arange(len(x_labels)), [str(state) for state in x_labels])
    plt.yticks(np.arange(len(y_labels)), [str(state) for state in y_labels])
    plt.tight_layout()  #
    plt.show()
    return

def probabilityCalcold(e1,e2,e3,e4): #calculate the probability of the states after doing the dot product (slay)

    n=2 #cause girl we got 2 photons
    m=6 #she be a 6 x 6 matrix right
    
    myfockbasis=basis(n,m)
    ind = [myfockbasis.index([0, 1, 0, 1, 0, 0]),
           myfockbasis.index([0, 1, 0, 0, 1, 0]),
           myfockbasis.index([0, 0, 1, 1, 0, 0]),
           myfockbasis.index([0, 0, 1, 0, 1, 0])] 
    print("fock state:[0, 1, 0, 1, 0, 0]")
    p1=np.abs(e1)**2
    p1array = np.array(p1)
    p1norm=np.linalg.norm(p1array)
    for i, prob in enumerate(p1):
        print(f"Probability of state {i}: {prob:.4f}")

    print("fock state:[0, 1, 0, 0, 1, 0]")
    p2=np.abs(e2)**2
    p2array = np.array(p2)
    p2norm=np.linalg.norm(p2array)
    for i, prob in enumerate(p2):
      print(f"Probability of state {i}: {prob:.4f}")

    print("fock state:[0, 0, 1, 1, 0, 0]")
    p3=np.abs(e3)**2
    p3array = np.array(p3)
    p3norm=np.linalg.norm(p3array)
    for i, prob in enumerate(p3):
        print(f"Probability of state {i}: {prob:.4f}")
    
    print("fock state:[0, 0, 1, 0, 1, 0]")
    p4=np.abs(e4)**2
    print(p4)
    p4array = np.array(p4)
    p4norm=np.linalg.norm(p4array)
    print(p4norm)
    for i, prob in enumerate(p4):
        print(f"Probability of state {i}: {prob:.4f}")
#I hope the code gods forgive my sins
    element11=np.dot(p1array,p1array)
    element12=np.dot(p1array,p2array)
    element13=np.dot(p1array,p3array)
    element14=np.dot(p1array,p4array)

    element21=np.dot(p2array,p1array)
    element22=np.dot(p2array,p2array)
    element23=np.dot(p2array,p3array)
    element24=np.dot(p2array,p4array)

    element31=np.dot(p3array,p1array)
    element32=np.dot(p3array,p2array)
    element33=np.dot(p3array,p3array)
    element34=np.dot(p3array,p4array)

    element41=np.dot(p4array,p1array)
    element42=np.dot(p4array,p2array)
    element43=np.dot(p4array,p3array)
    element44=np.dot(p4array,p4array)

    element11=np.dot(p1norm,p1norm)
    print(element11)
    '''element12=np.dot(p1norm,p2norm)
    element13=np.dot(p1norm,p3norm)
    element14=np.dot(p1norm,p4norm)

    element21=np.dot(p2norm,p1norm)
    element22=np.dot(p2norm,p2norm)
    element23=np.dot(p2norm,p3norm)
    element24=np.dot(p2norm,p4norm)

    element31=np.dot(p3norm,p1norm)
    element32=np.dot(p3norm,p2norm)
    element33=np.dot(p3norm,p3norm)
    element34=np.dot(p3norm,p4norm)

    element41=np.dot(p4norm,p1norm)
    element42=np.dot(p4norm,p2norm)
    element43=np.dot(p4norm,p3norm)
    element44=np.dot(p4norm,p4norm)'''

    full4by4= [[element11**2, element21**2, element31**2, element41**2], 
               [ element12**2,element22**2,element32**2,element42**2], 
               [element13**2,element23**2,element33**2,element43**2],
               [element14**2,element24**2,element34**2,element44**2]]
    #print(full4by4)



   # print(element1)

    myfockbasis = [
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0]]


    plt.figure() #oh wow much plot 
    hinton(np.real(full4by4), myfockbasis, myfockbasis)

    # Create a colorbar to indicate magnitudes
    # Create a colorbar to indicate magnitudes]    
    #hinton_plot = plt.imshow(np.real(full4by4), cmap='RdBu_r', interpolation='nearest')

    #norm = colors.Normalize(vmin=np.min(np.real(full4by4)), vmax=np.max(np.real(full4by4)))
    #bar = plt.colorbar(hinton_plot, ticks=[np.min(np.real(full4by4)), np.max(np.real(full4by4))], format='%0.2f')
    

    plt.xticks(np.arange(len(myfockbasis)), [str(state) for state in myfockbasis])
    plt.yticks(np.arange(len(myfockbasis)), [str(state) for state in myfockbasis])
    plt.show()


    return


def fockTest():
    n=2 #cause girl we got 2 photons
    m=6 #she be a 6 x 6 matrix right
    
    myfockbasis=basis(n,m)
    N=len(myfockbasis)
    ind = [myfockbasis.index([0, 1, 0, 1, 0, 0]),
           myfockbasis.index([0, 1, 0, 0, 1, 0]),
           myfockbasis.index([0, 0, 1, 1, 0, 0]),
           myfockbasis.index([0, 0, 1, 0, 1, 0])]  #these seem... alright?
    print(ind)
    psi1 = np.zeros((N), dtype=complex) # I think this is fine
    psi2 = np.zeros((N), dtype=complex) #there has got to be a better way but im sick and i dont care
    psi3 = np.zeros((N), dtype=complex)
    psi4 = np.zeros((N), dtype=complex)

    psi1[myfockbasis.index([0, 1, 0, 1, 0, 0])] = 1.0 #ive done the indexing correct here now
    psi2[myfockbasis.index([0, 1, 0, 0, 1, 0])] = 1.0
    psi3[myfockbasis.index([0, 0, 1, 1, 0, 0])] = 1.0
    psi4[myfockbasis.index([0, 0, 1, 0, 1, 0])] = 1.0

    multiguy=aa.multiPhotonUnitary(2, U)# I think this is fine

    e1=np.dot(multiguy, psi1) # Each element of wholething corresponds to the probability amplitude of the system being in a particular Fock basis state after the transformation.
    e2=np.dot(multiguy, psi2) #again, there should be a better way but im just going to make a ton of different arrays
    e3=np.dot(multiguy, psi3)
    e4=np.dot(multiguy, psi4)

    

    probabilityCalc(e1,e2,e3,e4,psi1,psi2,psi3,psi4) #sending this off to calculate the probabilites

    return 
fockTest()

def together():
    #were gonna try our best *positive vibes ONLY*
    multiguy=aa.multiPhotonUnitary(2, U) #reminder to make better variable names please
    fock=fockTest()

    return 



plt.plot 