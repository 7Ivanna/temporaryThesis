"""
The `quotonic.misc` module ...
"""

import numpy as np

from quotonic.fock import basis


def genHaarUnitary(m: int) -> np.ndarray:
    """Generate an $m\\times m$ unitary sampled randomly from the Haar measure.

    This function follows the procedure outlined in [F. Mezzadri, “How to
    generate random matrices from classical compact groups”, arXiv:math-ph/0609050v2
    (2007)](https://arxiv.org/abs/math-ph/0609050).

    Args:
        m: Dimension of the square $m \\times m$ unitary

    Returns:
        A 2D array storing the Haar random $m\\times m$ unitary
    """

    z = np.random.randn(m, m) + 1j * np.random.randn(m, m) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    Lambda = d / np.abs(d)
    U: np.ndarray = np.multiply(q, Lambda)
    return U


def compBasis(numPhotons: int) -> list:
    """Generate the computational basis for a given number of photons.

    INSERT DOCUMENTATION HERE
    ADD TEST

    Returns:
        List of computational basis states, where each state is a list of the qubit state in each respective qubit slot
    """

    # Compute the dimension of the computational basis
    dim = 2**numPhotons

    # Leverage the relation between computational basis states and binary to create the states
    CB = []
    for i in range(dim):
        state = [int(j) for j in format(i, "0" + str(numPhotons) + "b")]
        CB.append(state)

    return CB


def cb_to_fock(state_cb: list) -> list:
    """Convert a computational basis state to its corresponding representation in the Fock basis.

    INSERT DOCUMENTATION HERE, ASSUMES DUAL-RAIL ENCODING WITH NO MISSING QUBITS
    ADD TEST

    Returns:
        Converted Fock basis state, a list containing the number of photons in each respective optical mode
    """

    # Check the validity of the computational basis state
    assert max(state_cb) < 2, "The provided computational basis state is invalid."

    state_fb = []
    for q in state_cb:
        # |0>_CB = |10>
        if q == 0:
            state_fb.append(1)
            state_fb.append(0)
        # |1>_CB = |01>
        else:
            state_fb.append(0)
            state_fb.append(1)

    return state_fb


def fock_to_cb(state_fb: list) -> list:
    """Convert a Fock basis state to its corresponding representation in the computational basis.

    INSERT DOCUMENTATION HERE, ASSUMES DUAL-RAIL ENCODING WITH NO MISSING QUBITS
    ADD TEST

    Returns:
        Converted computational basis state, a list containing the qubit state in each respective qubit slot
    """

    # Check the validity of the Fock state
    assert max(state_fb) < 2, "The provided Fock basis state is invalid."

    state_cb = []
    for m in range(0, len(state_fb), 2):
        if state_fb[m] == 1 and state_fb[m + 1] == 0:
            state_cb.append(0)
        elif state_fb[m] == 0 and state_fb[m + 1] == 1:
            state_cb.append(1)

    # Check the validity of the generated computational basis state
    assert len(state_cb) == sum(state_fb), "The provided Fock basis state is invalid."

    return state_cb


def compBasisIndices(numPhotons: int, numModes: int) -> list:
    """Extract the indices of Fock basis states that correspond to computational basis states.

    INSERT DOCUMENTATION HERE
    ADD TEST

    Returns:
        List of the indices of the Fock basis states that correspond to computational basis states.
    """

    # Construct the Fock basis
    fockBasis = basis(numPhotons, numModes)

    CB_inds = []
    for s, state in enumerate(fockBasis):
        flag = False
        # Dual-rail encoding has at most one photon per optical mode
        if max(state) < 2:
            # Dual-rail encoding has at most one photon per consecutive pair of optical modes
            for m in range(0, numModes, 2):
                if state[m] == state[m + 1]:
                    flag = True
                    break
            if not flag:
                CB_inds.append(s)

    return CB_inds
