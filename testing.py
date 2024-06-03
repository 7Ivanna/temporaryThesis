import numpy as np

from quotonic.clements import Mesh
from quotonic.misc import genHaarUnitary


def test_Mesh():
    mesh = Mesh(4)

    assert mesh.numModes == 4
    
    assert mesh.alphaWG is None


def test_mzi():
    mesh = Mesh(2)

    result = np.array([[0, 1j], [1j, 0]], dtype=complex)
   #assert np.allclose(mesh.mzi(0, 0), result)

    result = np.conjugate(result).T
    #assert np.allclose(mesh.mzi(0, 0, inv=True), result)
    print(result)
print(test_mzi())

def test_mzi_column():
    mesh = Mesh(4)
    placementSpecifier = 0
    phis = np.zeros(2)
    twothetas = np.zeros(2)
    SRs = 0.5 * np.ones(4)
    alphas = np.zeros(4)

    result = np.array([[0, 1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, 1j, 0]], dtype=complex)
    assert np.allclose(mesh.mzi_column(placementSpecifier, phis, twothetas, SRs, alphas), result)
#print(test_mzi_column())
def test_haar_to_decode_to_encode():
    for m in range(3, 7):
        mesh = Mesh(m)
        for _ in range(100):
            U = genHaarUnitary(m)
            print(U)
            mesh.decode(U)
            assert np.allclose(mesh.encode(), U, atol=1e-4)
#print(test_haar_to_decode_to_encode())


def test_set_SR():
    mesh = Mesh(4)
    SR = 0.5 * np.ones(2)
    mesh.set_SR(SR)
    assert np.allclose(mesh.SR, SR)


def test_set_alpha():
    mesh = Mesh(4)
    alpha = 0.5 * np.ones(2)
    mesh.set_alpha(alpha)
    assert np.allclose(mesh.alpha, alpha)


def test_set_phases():
    mesh = Mesh(4)
    phases = 0.5 * np.ones(2)
    mesh.set_phases(phases)
    assert np.allclose(mesh.phases, phases)

"""
The `quotonic.training_sets` module ...
"""

import numpy as np

from quotonic.fock import basis, getDim


def BSA() -> tuple:
    """Construct QPNN-based BSA training set.

    INSERT DOCUMENTATION HERE
    ADD TEST

    Returns:
        Tuple of the number of input-output pairs, $K$, a 2D $N\\times K$ array of the input states resolved in the Fock basis, and a 2D $N\\times K$ array of the corresponding output states resolved in the Fock basis
    """
    # Set number of photons, n, and number of optical modes, m
    n = 2
    m = 4

    # Compute Fock basis dimension, N
    N = getDim(n, m)

    # Construct BSA training set
    K = 4

    psiIn = np.zeros((N, K), dtype=complex)
    psiIn[2, :] = np.array([1, 1, 0, 0], dtype=complex)  # |00>_CB = |1010>
    psiIn[3, :] = np.array([0, 0, 1, 1], dtype=complex)  # |01>_CB = |1001>
    psiIn[5, :] = np.array([0, 0, 1, -1], dtype=complex)  # |10>_CB = |0110>
    psiIn[6, :] = np.array([1, -1, 0, 0], dtype=complex)  # |11>_CB = |0101>
    psiIn /= np.sqrt(2.0)

    psiOut = np.zeros((N, K), dtype=complex)
    psiOut[2, :] = np.array([1, 0, 0, 0], dtype=complex)  # |00>_CB = |1010>
    psiOut[3, :] = np.array([0, 1, 0, 0], dtype=complex)  # |01>_CB = |1001>
    psiOut[5, :] = np.array([0, 0, 1, 0], dtype=complex)  # |10>_CB = |0110>
    psiOut[6, :] = np.array([0, 0, 0, 1], dtype=complex)  # |11>_CB = |0101>

    return K, psiIn, psiOut


def CNOT() -> tuple:
    """Construct QPNN-based CNOT gate training set.

    INSERT DOCUMENTATION HERE
    ADD TEST

    Returns:
        Tuple of the number of input-output pairs, $K$, a 2D $N\\times K$ array of the input states resolved in the Fock basis, and a 2D $N\\times K$ array of the corresponding output states resolved in the Fock basis
    """

    # Set number of photons, n, and number of optical modes, m
    n = 2
    m = 4

    # Compute Fock basis dimension, N
    N = getDim(n, m)
    print(N)
    # Construct CNOT training set
    K = 4

    psiIn = np.zeros((N, K), dtype=complex)
    psiIn[2, :] = np.array([1, 0, 0, 0], dtype=complex)  # |00>_CB = |1010>
    psiIn[3, :] = np.array([0, 1, 0, 0], dtype=complex)  # |01>_CB = |1001>
    psiIn[5, :] = np.array([0, 0, 1, 0], dtype=complex)  # |10>_CB = |0110>
    psiIn[6, :] = np.array([0, 0, 0, 1], dtype=complex)  # |11>_CB = |0101>

    psiOut = np.zeros((N, K), dtype=complex)
    psiOut[2, :] = np.array([1, 0, 0, 0], dtype=complex)  # |00>_CB = |1010>
    psiOut[3, :] = np.array([0, 1, 0, 0], dtype=complex)  # |01>_CB = |1001>
    psiOut[5, :] = np.array([0, 0, 0, 1], dtype=complex)  # |10>_CB = |0110>
    psiOut[6, :] = np.array([0, 0, 1, 0], dtype=complex)  # |11>_CB = |0101>

    return K, psiIn, psiOut
CNOT()

def CZ() -> tuple:
    """Construct QPNN-based CZ gate training set.

    INSERT DOCUMENTATION HERE
    ADD TEST

    Returns:
        Tuple of the number of input-output pairs, $K$, a 2D $N\\times K$ array of the input states resolved in the Fock basis, and a 2D $N\\times K$ array of the corresponding output states resolved in the Fock basis
    """

    # Set number of photons, n, and number of optical modes, m
    n = 2
    m = 4

    # Compute Fock basis dimension, N
    N = getDim(n, m)
    print(N)
    # Construct CZ training set
    K = 4

    psiIn = np.zeros((N, K), dtype=complex)
    psiIn[2, :] = np.array([1, 0, 0, 0], dtype=complex)  # |00>_CB = |1010>
    psiIn[3, :] = np.array([0, 1, 0, 0], dtype=complex)  # |01>_CB = |1001>
    psiIn[5, :] = np.array([0, 0, 1, 0], dtype=complex)  # |10>_CB = |0110>
    psiIn[6, :] = np.array([0, 0, 0, 1], dtype=complex)  # |11>_CB = |0101>

    psiOut = np.zeros((N, K), dtype=complex)
    psiOut[2, :] = np.array([1, 0, 0, 0], dtype=complex)  # |00>_CB = |1010>
    psiOut[3, :] = np.array([0, 1, 0, 0], dtype=complex)  # |01>_CB = |1001>
    psiOut[5, :] = np.array([0, 0, -1, 0], dtype=complex)  # |10>_CB = |0110>
    psiOut[6, :] = np.array([0, 0, 0, -1], dtype=complex)  # |11>_CB = |0101>

    return K, psiIn, psiOut


def GHZ() -> tuple:
    """Construct QPNN-based GHZ generator training set.

    INSERT DOCUMENTATION HERE
    ADD TEST

    Returns:
        Tuple of the number of input-output pairs, $K$, a 2D $N\\times K$ array of the input states resolved in the Fock basis, and a 2D $N\\times K$ array of the corresponding output states resolved in the Fock basis
    """

    # Set number of photons, n, and number of optical modes, m
    n = 3
    m = 6

    # Get Fock basis and index the nonzero components in psiIn, psiOut
    fockBasis = basis(n, m)
    nonzeroComponents = [fockBasis.index([1, 0, 1, 0, 1, 0]), fockBasis.index([0, 1, 0, 1, 0, 1])]
    N = len(fockBasis)
    print(N)
    print(nonzeroComponents)
    # Construct GHZ training set
    K = 1

    psiIn = np.zeros((N, K), dtype=complex)
    psiIn[nonzeroComponents[0], 0] = 1.0

    psiOut = np.zeros((N, K), dtype=complex)
    psiOut[nonzeroComponents[0], 0] = 1 / np.sqrt(2.0)
    psiOut[nonzeroComponents[1], 0] = 1 / np.sqrt(2.0)

    return K, psiIn, psiOut
GHZ() 