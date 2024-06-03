from quotonic.clements import Mesh
import numpy as np
import quotonic.aa as aa
from quotonic.fock import basis, getDim
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt



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
    #plt.tight_layout()  #

def probabilityCalc(e1,e2,e3,e4,psi1,psi2,psi3,psi4):
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

    full4by4= np.square(np.abs(np.array([[element11, element21, element31, element41], 
               [ element12,element22,element32,element42], 
               [element13,element23,element33,element43],
               [element14,element24,element34,element44]])))
    myfockbasis = [
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0]]
    plt.figure()
    y_labels = ['⟨00|','⟨01|', '⟨10|', '⟨11|']
    x_labels = ['|00⟩','|01⟩', '|10⟩', '|11⟩']
    hinton(np.real(full4by4), x_labels, y_labels)
    plt.xticks(np.arange(len(x_labels)), [str(state) for state in x_labels])
    plt.yticks(np.arange(len(y_labels)), [str(state) for state in y_labels])
    plt.show()
    return

def unconditional():
    N=6
    n=2 #cause girl we got 2 photons
    m=6 #she be a 6 x 6 matrix right
    alpha_mzi = 0.01145
    alpha_fibre = 0.0000460506
    total_lenth =  1.4727148750970904 #m
    mzi_loss_even = (np.sqrt(1-alpha_mzi))**N # loss for even modes
    mzi_loss_odd = (np.sqrt(1-alpha_mzi))**(N+1) # loss for odd modes
    fibre_loss =  (np.sqrt(1-alpha_fibre))**(total_lenth)

    D=np.array( #the D array
            [
                [mzi_loss_odd*fibre_loss, 0,0,0,0,0],
                [0,mzi_loss_even*fibre_loss,0,0,0,0],
                [0,0,mzi_loss_odd*fibre_loss,0,0,0],
                [0,0,0,mzi_loss_even*fibre_loss,0,0],
                [0,0,0,0,mzi_loss_odd*fibre_loss,0],
                [0,0,0,0,0,mzi_loss_even*fibre_loss]
            ],dtype=complex)
    mesh = Mesh(6)
    U = np.array([[-1, np.sqrt(2), 0, 0, 0, 0],
            [np.sqrt(2), 1, 0, 0, 0, 0],
            [0, 0, -1, 1, 1, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, -1],
            [0, 0, 0, 1, -1, -1]], dtype=complex) / np.sqrt(3)
    print(mesh.decode(U))
 
    print("the phases are", mesh.phases)
    
    new=np.dot(D,U)

    multiguy=aa.multiPhotonUnitary(2, new)# I think this is fine
    # multiguy = np.square(np.abs(multiguy))
    myfockbasis=basis(n,m)
    N=len(myfockbasis)
    ind = [myfockbasis.index([0, 1, 0, 1, 0, 0]),
           myfockbasis.index([0, 1, 0, 0, 1, 0]),
           myfockbasis.index([0, 0, 1, 1, 0, 0]),
           myfockbasis.index([0, 0, 1, 0, 1, 0])]  #these seem... alright?
    
    print(multiguy[ind[0], ind[0]], multiguy[ind[1], ind[1]], multiguy[ind[2], ind[3]], multiguy[ind[3], ind[2]])

    psi1 = np.zeros((N), dtype=complex) # I think this is fine
    psi2 = np.zeros((N), dtype=complex) #there has got to be a better way but im sick and i dont care
    psi3 = np.zeros((N), dtype=complex)
    psi4 = np.zeros((N), dtype=complex)

    psi1[myfockbasis.index([0, 1, 0, 1, 0, 0])] = 1.0 #ive done the indexing correct here now
    psi2[myfockbasis.index([0, 1, 0, 0, 1, 0])] = 1.0
    psi3[myfockbasis.index([0, 0, 1, 1, 0, 0])] = 1.0
    psi4[myfockbasis.index([0, 0, 1, 0, 1, 0])] = 1.0

    e1=np.dot(multiguy, psi1) # Each element of wholething corresponds to the probability amplitude of the system being in a particular Fock basis state after the transformation.
    e2=np.dot(multiguy, psi2) #again, there should be a better way but im just going to make a ton of different arrays
    e3=np.dot(multiguy, psi3)
    e4=np.dot(multiguy, psi4)


    probabilityCalc(e1,e2,e3,e4,psi1,psi2,psi3,psi4) #sending this off to calculate the probabilites

    return 
unconditional()


def conditionalhinton(matrix, x_labels=None, y_labels=None, max_weight=None, ax=None): #hinton timeeeeeeeeeee
    ax = ax if ax is not None else plt.gca()
    if not max_weight:
        max_weight = np.max(np.abs(matrix))

    # Set the background color of the axes
    ax.set_facecolor('white')
    ax.set_title('CNOT Gate Results for a 2 Photon System Conditional Fidelity')
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

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation='vertical')

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.5, color='black')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability')

def conditionalprob(e1,e2,e3,e4,psi1,psi2,psi3,psi4):
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

    full4by4= 1/21*np.square(np.abs(np.array([[element11, element21, element31, element41], 
               [ element12,element22,element32,element42], 
               [element13,element23,element33,element43],
               [element14,element24,element34,element44]])))
    myfockbasis = [
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0]]
    plt.figure()
    y_labels = ['⟨00|','⟨01|', '⟨10|', '⟨11|']
    x_labels = ['|00⟩','|01⟩', '|10⟩', '|11⟩']
    conditionalhinton(np.real(full4by4), x_labels, y_labels)
    plt.xticks(np.arange(len(x_labels)), [str(state) for state in x_labels])
    plt.yticks(np.arange(len(x_labels)), [str(state) for state in x_labels])
    plt.show()
    return


def conditional():
    N=6
    n=2 #cause girl we got 2 photons
    m=6 #she be a 6 x 6 matrix right
    alpha_mzi =  0.01145
    alpha_fibre = 0.0000460506
    total_lenth =  1.4727148750970904 #m
    mzi_loss_even = (np.sqrt(1-alpha_mzi))**N # loss for even modes
    mzi_loss_odd = (np.sqrt(1-alpha_mzi))**(N+1) # loss for odd modes
    fibre_loss =  (np.sqrt(1-alpha_fibre))**(total_lenth)

    D=np.array( #the D array
            [
                [mzi_loss_odd*fibre_loss, 0,0,0,0,0],
                [0,mzi_loss_odd*fibre_loss,0,0,0,0],
                [0,0,mzi_loss_odd*fibre_loss,0,0,0],
                [0,0,0,mzi_loss_even*fibre_loss,0,0],
                [0,0,0,0,mzi_loss_even*fibre_loss,0],
                [0,0,0,0,0,mzi_loss_even*fibre_loss]
            ],dtype=complex)
    mesh = Mesh(6)
    U = np.array([[-1, np.sqrt(2), 0, 0, 0, 0],
            [np.sqrt(2), 1, 0, 0, 0, 0],
            [0, 0, -1, 1, 1, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, -1],
            [0, 0, 0, 1, -1, -1]], dtype=complex) / np.sqrt(3)
    print(mesh.decode(U))
 
    print("the phases are", mesh.phases)
    
    new=np.dot(D,U)

    new=np.dot(D,U)

    multiguy=aa.multiPhotonUnitary(2, new)# I think this is fine
    # multiguy = np.square(np.abs(multiguy)) I think this is fine
    myfockbasis=basis(n,m)
    N=len(myfockbasis)
    ind = [myfockbasis.index([0, 1, 0, 1, 0, 0]),
           myfockbasis.index([0, 1, 0, 0, 1, 0]),
           myfockbasis.index([0, 0, 1, 1, 0, 0]),
           myfockbasis.index([0, 0, 1, 0, 1, 0])]  #these seem... alright?

    psi1 = np.zeros((N), dtype=complex) # I think this is fine
    psi2 = np.zeros((N), dtype=complex) #there has got to be a better way but im sick and i dont care
    psi3 = np.zeros((N), dtype=complex)
    psi4 = np.zeros((N), dtype=complex)

    psi1[myfockbasis.index([0, 1, 0, 1, 0, 0])] = 1.0 #ive done the indexing correct here now
    psi2[myfockbasis.index([0, 1, 0, 0, 1, 0])] = 1.0
    psi3[myfockbasis.index([0, 0, 1, 1, 0, 0])] = 1.0
    psi4[myfockbasis.index([0, 0, 1, 0, 1, 0])] = 1.0

    e1=np.dot(multiguy, psi1) # Each element of wholething corresponds to the probability amplitude of the system being in a particular Fock basis state after the transformation.
    e2=np.dot(multiguy, psi2) #again, there should be a better way but im just going to make a ton of different arrays
    e3=np.dot(multiguy, psi3)
    e4=np.dot(multiguy, psi4)


    conditionalprob(e1,e2,e3,e4,psi1,psi2,psi3,psi4) #sending this off to calculate the probabilites
conditional()



# %%
