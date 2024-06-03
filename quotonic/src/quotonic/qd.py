"""
The `quotonic.qd` modules includes ...
"""

from typing import Optional

import numpy as np

import quotonic.fock as fock


def complex_exp(x: float) -> complex:
    """Wrapper for computing complex exponentials $e^{ix}$, designed to speed up the calculation for $x = 0$ and $x = \\pi$.

    Args:
        x: Argument of the complex exponential

    Returns:
        Result of the complex exponential, $e^{ix}$
    """

    # Simplify calculation of e^{i(0)} = 1
    if x == 0:
        return complex(1)
    # Simplify calculation of e^{i(\pi)} = -1
    elif x == np.pi:
        return complex(-1)
    return complex(np.exp(1j * x))


def buildQDUnitary(numPhotons: int, numModes: int, varphi: float, burnoutMap: Optional[np.ndarray] = None) -> np.ndarray:
    """Construct the diagonal unitary nonlinear QD matrix.

    INSERT DOCUMENTATION HERE

    Args:
        numPhotons: Number of photons, $n$
        numModes: Number of optical modes, $m$
        varphi: Effective nonlinear phase shift in $\\text{rad}$, $\\varphi$
        burnoutMap: A 1D array of length $m$, with either boolean or binary elements, specifying whether nonlinearities are on/off for specific modes

    Returns:
        A 2D $N\\times N$ array, the matrix representation of the set of single-site QD nonlinearities resolved in the Fock basis
    """

    # Check if burnoutMap has been provided, otherwise, choose default (all nonlinearities applied)
    if burnoutMap is None:
        burnoutMap = np.ones(numModes)

    # Retrieve Fock basis for the given numbers of photons and optical modes
    fockBasis = fock.basis(numPhotons, numModes)
    fockDim = len(fockBasis)

    # Initialize the diagonal of the QD unitary \Sigma(\phi)
    Sigma = np.ones(fockDim, dtype=complex)

    for i, fockState in enumerate(fockBasis):
        QDPhase = 0.0
        # For each Fock basis state, sum the phase shifts from each optical mode
        for mode in range(numModes):
            if fockState[mode] > 0 and burnoutMap[mode] == 1:
                QDPhase += varphi

        Sigma[i] = complex_exp(QDPhase)

    # Return fockDim x fockDim diagonal QD unitary matrix
    return np.diag(Sigma)
