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

    # Construct GHZ training set
    K = 1

    psiIn = np.zeros((N, K), dtype=complex)
    psiIn[nonzeroComponents[0], 0] = 1.0

    psiOut = np.zeros((N, K), dtype=complex)
    psiOut[nonzeroComponents[0], 0] = 1 / np.sqrt(2.0)
    psiOut[nonzeroComponents[1], 0] = 1 / np.sqrt(2.0)

    return K, psiIn, psiOut
