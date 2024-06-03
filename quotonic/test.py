from quotonic.clements import Mesh
import numpy as np
from quotonic.misc import genHaarUnitary
import math as m
def testcolumn():
    mesh = Mesh(4)
    placementSpecifier = 0#this represents the column you're looking at? , else, i dont know what is meant by being inserted 
    phis = np.array([0,-1]) #can you not pass pi here? I'll assume no for now looking at the clements code, but ask to make sure 
    alphas = np.zeros(4)
    twothetas = np.zeros(2)
    SRs = 0.5*np.ones(4)
    

    result = np.array([[0, 1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]], dtype=complex)
    assert np.allclose(mesh.mzi_column(placementSpecifier, phis, twothetas, SRs, alphas), result)
print(testcolumn())'

def three_column_test(): #just to see if i really get it
    mesh = Mesh(3)
    placementSpecifier = 0
    phis =  np.array([0,np.pi]) 
    twothetas = np.zeros(1)
    SRs = 0.5 * np.ones(3)
    alphas = np.zeros(3)
    result_matrix = mesh.mzi_column(placementSpecifier, phis, twothetas, SRs, alphas)
    print(result_matrix)
    result = np.array([[0, 1j, 0], [1j, 0, 0], [0, 0, 1]], dtype=complex)
    assert np.allclose(mesh.mzi_column(placementSpecifier, phis, twothetas, SRs, alphas), result)
    return result_matrix
#print( three_column_test())


# Print the result_matrix




def encodingtest():
     mesh = Mesh(3)
     mesh.numModes=3
     phases = np.array([np.pi/2,np.pi,np.pi/2,np.pi,np.pi/2,np.pi,0,0,0])

     mesh.set_phases(phases)
   
     U=mesh.encode()
     print("decode next")
     print(mesh.phases)

     assert np.allclose(mesh.phases, phases)
     #print(mesh.encode())
     print("decode next")
    # print(mesh.decode(U))
print(encodingtest())

def decodingtest():
    mesh=Mesh(2)
    U=np.array([[1j,-1j],[-1j,1j]])
    print(mesh.decode(U))
#print(decodingtest())


