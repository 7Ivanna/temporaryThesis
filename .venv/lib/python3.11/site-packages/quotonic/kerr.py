"""
The `quotonic.kerr` modules includes functions required to construct the transfer matrix of a specified
set of single-site optical Kerr nonlinearities across $m$ optical modes.

All code in this module is adapted from [Bosonic: A Quantum Optics Library](https://github.com/steinbrecher/bosonic),
as originally designed for use in [G. R. Steinbrecher *et al*., “Quantum optical neural networks”,
*npj Quantum Inf* **5**, 60 (2019)](https://doi.org/10.1038/s41534-019-0174-7).
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


def buildKerrUnitary(numPhotons: int, numModes: int, varphi: float, burnoutMap: Optional[np.ndarray] = None) -> np.ndarray:
    """Construct the diagonal unitary nonlinear Kerr matrix.

    This function constructs the diagonal unitary matrix, $\\boldsymbol{\\Sigma}(\\varphi)$,
    corresponding to single-site Kerr nonlinearities of strength $\\varphi$ applied across a
    set of $m$ optical modes. If all single-site nonlinearities are applied, then this matrix
    is expressed mathematically as,

    $$ \\boldsymbol{\\Sigma}(\\varphi) = \\sum_{n=0}^\\infty e^{in(n-1)\\frac{\\varphi}{2}}\\left|n\\right\\rangle\\left\\langle n\\right|. $$

    From this form, it is evident that $\\boldsymbol{\\Sigma}(\\varphi)$ is the $N\\times N$
    identity matrix ($\\mathbf{I}_N$), where $N$ is the Fock basis dimension, in cases of
    $n < 2$, where $n$ is the number of photons. The $n$ is the previous expression is better
    explained as the number of photons in the optical mode for which a single-site optical Kerr
    nonlinearity is applied. This is best displayed by example. In the case where $n=3$, $m=2$,
    the Fock basis has a dimension of $N = 4$ with basis states $\\left\\{\\left|30\\right\\rangle,
    \\left|21\\right\\rangle, \\left|12\\right\\rangle, \\left|03\\right\\rangle\\right\\}$. As
    evident from the expression above, given the orthonormality of the Fock basis, the only
    nonzero elements lie on the diagonal of the unitary $\\boldsymbol{\\Sigma}(\\varphi)$.
    Consider the calculation of the element located at the row and column both corresponding to
    state $\\left|21\\right\\rangle$,

    $$ \\left\\langle 21 \\right|(\\boldsymbol{\\Sigma}(\\varphi)\\otimes\\mathbf{I}_N)(\\mathbf{I}_N\\otimes\\boldsymbol{\\Sigma}(\\varphi))\\left| 21 \\right\\rangle =
    \\left[\\sum_{n=0}^\\infty e^{in(n-1)\\frac{\\varphi}{2}}\\left\\langle 2|n \\right\\rangle\\left\\langle n|2 \\right\\rangle\\right]
    \\left[\\sum_{n=0}^\\infty e^{in(n-1)\\frac{\\varphi}{2}}\\left\\langle 1|n \\right\\rangle\\left\\langle n|1 \\right\\rangle\\right]
    = \\left[e^{i\\varphi}\\right]\\left[e^{i(0)}\\right] = e^{i\\varphi}. $$

    From the tensor products, it is evident that the Kerr nonlinearities are single-site and thus
    applied by mode. After completing the calculation of the other elements along the diagonal,
    the resulting unitary is given as follows,

    $$ \\boldsymbol{\\Sigma}(\\varphi) = \\begin{pmatrix} e^{i3\\varphi} & 0 & 0 & 0 \\\ 0 & e^{i\\varphi} & 0 & 0 \\\ 0 & 0 & e^{i\\varphi} & 0 \\\ 0 & 0 & 0 & e^{i3\\varphi} \\end{pmatrix}, $$

    where the ordering of the rows and columns follows that in which the basis states were listed
    previously. The previous process can be modified by *burning out* some of the single-site
    nonlinearities. This is conducted by passing an array of binary/boolean values to the function
    in `burnoutMap`. This 1D array has $m$ elements, each respectively telling the function
    whether the single-site nonlinearity at a specific mode is on or off.

    Args:
        numPhotons: Number of photons, $n$
        numModes: Number of optical modes, $m$
        varphi: Effective nonlinear phase shift in $\\text{rad}$, $\\varphi$
        burnoutMap: A 1D array of length $m$, with either boolean or binary elements, specifying whether nonlinearities are on/off for specific modes

    Returns:
        A 2D $N\\times N$ array, the matrix representation of the set of single-site Kerr nonlinearities resolved in the Fock basis
    """

    # Check if burnoutMap has been provided, otherwise, choose default (all nonlinearities applied)
    if burnoutMap is None:
        burnoutMap = np.ones(numModes)

    # Retrieve Fock basis for the given numbers of photons and optical modes
    fockBasis = fock.basis(numPhotons, numModes)
    fockDim = len(fockBasis)

    # Initialize the diagonal of the Kerr unitary \Sigma(\phi)
    Sigma = np.ones(fockDim, dtype=complex)

    # If the number of photons is 0 or 1, then the Kerr unitary is an identity matrix
    if numPhotons < 2:
        return np.diag(Sigma)

    for i, fockState in enumerate(fockBasis):
        kerrPhase = 0
        # For each Fock basis state, sum the phase shifts from each optical mode
        for mode in range(numModes):
            if fockState[mode] > 1 and burnoutMap[mode] == 1:
                kerrPhase += fockState[mode] * (fockState[mode] - 1) * varphi * 0.5

        Sigma[i] = complex_exp(kerrPhase)

    # Return fockDim x fockDim diagonal Kerr unitary matrix
    return np.diag(Sigma)
