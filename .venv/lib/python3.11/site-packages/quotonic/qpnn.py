"""
The `quotonic.qpnn` module includes a class that allows quantum photonic neural networks
(QPNNs) to be instantiated. When instantiating a QPNN, the user can specify whether it
should include photon propagation losses (balanced or unbalanced) and/or imperfect
directional coupler splitting ratios. The user may also specify the strength of the
network's single-site Kerr nonlinearities, and/or which ones are on/off.

The user's specifications are passed through to other classes/functions when constructing
the network model. A QPNN is designed to operate on a certain number of photons, $n$, with
a certain number of optical modes $m$. It also features $L$ layers where each is a mesh of
Mach-Zehnder interferometers (MZIs) arranged in the Clements configuration (see
[Clements](clements.md)). Single-site Kerr nonlinearities are wedged between layers
(see [Kerr](kerr.md)). Altogether, the system function is representative of a nonlinear
optical transformation on quantum photonic states that is parameterized by the phase shifts
in each MZI throughout the entire network.

Some code in this module is adapted from [Bosonic: A Quantum Optics Library](https://github.com/steinbrecher/bosonic),
as originally designed for use in [G. R. Steinbrecher *et al*., “Quantum optical neural networks”,
*npj Quantum Inf* **5**, 60 (2019)](https://doi.org/10.1038/s41534-019-0174-7). In this article,
and in [J. Ewaniuk *et al*., "Imperfect Quantum Photonic Neural Networks", *Adv Quantum
Technol.* 2200125 (2023)](https://doi.org/10.1002/qute.202200125), QPNNs are more
thoroughly described.
"""

from typing import Optional

import numpy as np

from quotonic.aa import multiPhotonUnitary
from quotonic.clements import Mesh
from quotonic.fock import getDim
from quotonic.kerr import buildKerrUnitary
from quotonic.misc import compBasisIndices
from quotonic.qd import buildQDUnitary


class QPNN:
    """Model of a quantum photonic neural network (QPNN).

    Each QPNN is designed to operate on a certain number of photons, $n$ with a certain
    number of optical modes $m$, and features $L$ layers. The user may also specify the
    imperfections experienced by a given `QPNN` instance. The options include photon
    propagation losses (balanced or unbalanced), imbalanced directional coupler
    splitting ratios, weak optical nonlinearities, or even *burnt out* nonlinearities.

    This class allows QPNNs to be instantiated with provided characteristics. Then, the
    system transfer function of the network may be evaluated based on the provided or
    default attributes.

    Attributes:
        numPhotons (int): Number of photons, $n$
        numModes (int): Number of optical modes, $m$
        numLayers (int): Number of layers, $L$
        varphi (float): Effective nonlinear phase shift, $\\varphi$
        burnoutMap (Optional[np.ndarray]): A 1D array of length $m$, with either boolean or binary elements, specifying whether nonlinearities are on/off for specific modes
        alphaWG (Optional[float]): Mean propagation losses, $\\alpha_\\mathrm{WG}$, in $\\text{dB}/\\text{cm}$
        std_alphaWG (Optional[float]): Standard deviation of the propagation losses, $\\alpha_\\mathrm{WG}$, in $\\text{dB}/\\text{cm}$
        std_SR (Optional[float]): Standard deviation of the directional coupler splitting ratio, $t$
        ellMZI (float): Characteristic length of a MZI, $\\ell_\\mathrm{MZI}$, in $\\text{cm}$
        ellPS (float): Characteristic length of a phase shifter, $\\ell_\\mathrm{PS}$, in $\\text{cm}$
        ellF (float): Characteristic length of a flat section in parallel with a MZI, $\\ell_\\mathrm{F}$, in $\\text{cm}$
        alphas (list[np.ndarray]): List of 1D arrays of the fractions of light lost in each component throughout the network, separated by layer
        SRs (list[np.ndarray]): List of 1D arrays of the directional coupler splitting ratios throughout the network, separated by layer
        fockDim (int): Dimension of the Fock basis, $N$
        pspl (int): Number of phase shift parameters per QPNN layer, $m^2$
        allPhases (np.ndarray): A 1D array of length $m^2L$ storing all the phase shift parameters for the network
        mesh (list): A list of all the Clements MZI meshes in the QPNN, one mesh per layer
        nlU (list): A list of all the single-site nonlinearity sections in the QPNN, one between each pair of layers
        S (np.ndarray): QPNN system transfer function, $\\mathbf{S}$
        K (Optional[int]): The number of input-output state pairs in the QPNN training set
        psiIn (np.ndarray): An $N\\times K$ 2D array containing the $K$ input states of the QPNN training set, resolved in the $N$-dimensional Fock basis
        psiOut (np.ndarray): An $N\\times K$ 2D array containing the $K$ output states of the QPNN training set, resolved in the $N$-dimensional Fock basis
        Func (float): The unconditional fidelity, $\\mathcal{F}^\\mathrm{(unc)}$, of the QPNN
        Fcon (float): The conditional fidelity, $\\mathcal{F}^\\mathrm{(con)}$, of the QPNN
        Pcb (float): The computational basis probability, $\\mathcal{P}^\\mathrm{(cb)}$, of the QPNN
        compBasisInds (list): List of the indices of states in the Fock basis that correspond to the computational basis
        compBasisDim (int): Dimension of the computational basis
    """

    def __init__(
        self,
        numPhotons: int,
        numModes: int,
        numLayers: int,
        nl: str = "kerr",
        varphi: float = np.pi,
        burnoutMap: Optional[np.ndarray] = None,
        alphaWG: Optional[float] = None,
        std_alphaWG: Optional[float] = None,
        std_SR: Optional[float] = None,
        ellMZI: float = 0.028668,
        ellPS: float = 0.0050,
        ellF: float = 0.028668,
        trainingSet: Optional[tuple] = None,
    ) -> None:
        """Initialization of a QPNN instance.

        The characteristics of the QPNN are first stored, and others are computed from those
        provided. Then, the specifications are used to construct the pieces of the network,
        layer by layer. For each layer, a Clements MZI mesh is instantiated, and any losses
        or imperfect directional coupler splitting ratios for each mesh are stored in the
        attributes of the QPNN instance. Between pairs of layers, single-site nonlinearity
        functions are built according to the provided effective nonlinear phase shift,
        $\\varphi$, and `burnoutMap`.

        Args:
            numPhotons: Number of photons, $n$
            numModes: Number of optical modes, $m$
            numLayers: Number of layers, $L$
            nl: Type of single-site nonlinearity, $\\Sigma(\\varphi)$
            varphi: Effective nonlinear phase shift, $\\varphi$
            burnoutMap: A 1D array of length $m$, with either boolean or binary elements, specifying whether nonlinearities are on/off for specific modes
            alphaWG: Mean propagation losses, $\\alpha_\\mathrm{WG}$, in $\\text{dB}/\\text{cm}$
            std_alphaWG: Standard deviation of the propagation losses, $\\alpha_\\mathrm{WG}$, in $\\text{dB}/\\text{cm}$
            std_SR: Standard deviation of the directional coupler splitting ratio, $t$
            ellMZI: Characteristic length of a MZI, $\\ell_\\mathrm{MZI}$, in $\\text{cm}$
            ellPS: Characteristic length of a phase shifter, $\\ell_\\mathrm{PS}$, in $\\text{cm}$
            ellF: Characteristic length of a flat section in parallel with a MZI, $\\ell_\\mathrm{F}$, in $\\text{cm}$
            trainingSet: A tuple including the number of input-output pairs, $K$, a 2D $N\\times K$ array of the input states resolved in the Fock basis, and a 2D $N\\times K$ array of the corresponding output states resolved in the Fock basis
        """

        # Store the provided properties of the QPNN, calculate others
        self.numPhotons = numPhotons
        self.numModes = numModes
        self.numLayers = numLayers
        self.fockDim = getDim(numPhotons, numModes)
        self.pspl = numModes * numModes
        self.alphaWG = alphaWG
        self.std_alphaWG = std_alphaWG
        self.std_SR = std_SR
        self.ellMZI = ellMZI
        self.ellPS = ellPS
        self.ellF = ellF
        self.varphi = varphi
        self.burnoutMap = burnoutMap
        if burnoutMap is None:
            burnoutMap = np.ones((numModes, numLayers - 1))
        if trainingSet is None:
            self.K = None
            self.psiIn = np.zeros(1)
            self.psiOut = np.zeros(1)
        else:
            self.K = trainingSet[0]
            self.psiIn = trainingSet[1]
            self.psiOut = trainingSet[2]
        assert nl == "kerr" or nl == "qd", "Single-site nonlinearities must be kerr or qd."

        # Initialize phases, S, Func, Fcon, Pcb
        self.allPhases = np.zeros(numModes * numModes * numLayers)
        self.S = np.eye(self.fockDim)
        self.Func = 0.0
        self.Fcon = 0.0
        self.Pcb = 0.0

        # Create the network from MZI meshes and single-site nonlinearities
        self.mesh = []
        self.alphas = []
        self.SRs = []
        self.nlU = []
        for L in range(numLayers):
            self.mesh.append(
                Mesh(
                    numModes,
                    alphaWG=alphaWG,
                    std_alphaWG=std_alphaWG,
                    std_SR=std_SR,
                    ellMZI=ellMZI,
                    ellPS=ellPS,
                    ellF=ellF,
                )
            )
            # Store the losses and DC splitting ratios generated when instantiating the meshes
            self.alphas.append(self.mesh[L].alpha)
            self.SRs.append(self.mesh[L].SR)
            if L < numLayers - 1:
                if nl == "kerr":
                    self.nlU.append(buildKerrUnitary(numPhotons, numModes, varphi, burnoutMap[:, L]))
                elif nl == "qd":
                    self.nlU.append(buildQDUnitary(numPhotons, numModes, varphi, burnoutMap[:, L]))

        # Extract the indices of the Fock basis states that correspond to the computational basis
        if numModes % 2 == 0:
            self.compBasisInds = compBasisIndices(numPhotons, numModes)
        else:
            self.compBasisInds = []
        self.compBasisDim = len(self.compBasisInds)

    def set_phases(self, allPhases: np.ndarray) -> None:
        """Set all phase shifts in the QPNN.

        A QPNN features $L$ layers, and thus $L$ Clements MZI meshes, each
        with $m$ optical modes and $m^2$ phase shifts. Out of the total $m^2L$
        phase shifts, input, they are separated by layer and passed to the
        meshes that were instantiated for the network model upon initialization.

        Args:
            allPhases: A 1D array of length $m^2L$ with all the phase shift parameters for the network
        """

        # Separate the phases by layer, then add to respective mesh
        self.allPhases = allPhases
        for L in range(self.numLayers):
            self.mesh[L].set_phases(allPhases[L * self.pspl : (L + 1) * self.pspl])

    def set_alphas(self, alphas: list) -> None:
        """Set all component losses in the QPNN.

        A QPNN features $L$ layers, and thus $L$ Clements MZI meshes, each
        with $m$ optical modes, $\\frac{1}{2}m(m-1)$ MZIs, $m$ flat sections
        in parallel with MZIs, and $m$ output phase shifters. Each component
        contributes a specific fraction of light lost when photons propagate
        through them. The MZIs may have imbalanced losses in each arm, and
        thus there are two values for the fraction of light lost per MZI.
        Here, all of the fractions can be specified manually by inputting a
        list of lists of $\\alpha$ values for each layer. These are then
        passed to the corresponding meshes that were instantiated for the
        network model upon initialization.

        Args:
            alphas: An $L$-element list of 1D arrays of length $m(m-1) + 2m$ that respectively provide the fractions of light lost contributed by each component in each layer of the QPNN
        """

        # Separate the alphas by layer, then add to respective mesh
        for L in range(self.numLayers):
            self.mesh[L].set_alpha(alphas[L])
        self.alphas = alphas

    def set_SRs(self, SRs: list) -> None:
        """Set all directional coupler splitting ratios in the QPNN.

        A QPNN features $L$ layers, and thus $L$ Clements MZI meshes, each
        with $m$ optical modes, $\\frac{1}{2}m(m-1)$ MZIs, and thus $m(m-1)$
        directional couplers. Each directional coupler may have a specific
        splitting ratio. Here, each ratio can be specified manually by
        inputting a list of lists of $t$ values for each layer. These are
        then passed to the corresponding meshes that were instantiated for
        the network model.

        Args:
            SRs: An $L$-element list of 1D arrays of length $m(m-1)$ that respectively provide the splitting ratios, $t$, for each directional coupler in each layer of the QPNN
        """

        # Separate the splitting ratios by layer, then add to respective mesh
        for L in range(self.numLayers):
            self.mesh[L].set_SR(SRs[L])
        self.SRs = SRs

    def sysFunc(self) -> np.ndarray:
        """Compute the system transfer function, $\\mathbf{S}$, of the QPNN.

        An example QPNN, with two layers, two photons, and four optical modes,
        is shown below. This function computes the matrix representation of the
        QPNN, like the one shown below, from the pieces of the architecture.

        <p align="center">
        <img width="800" src="img/qpnn.png">
        </p>

        Mathematically, the system function is given by,

        $$ \\mathbf{S} = \\mathbf{U}(\\boldsymbol{\\phi}_L, \\boldsymbol{\\theta}_L) \\cdot
        \\prod_{i=1}^{L-1}\\boldsymbol{\\Sigma}(\\varphi) \\cdot
        \\mathbf{U}(\\boldsymbol{\\phi}_i, \\boldsymbol{\\theta}_i), $$

        where $\\mathbf{U}(\\boldsymbol{\\phi}_i, \\boldsymbol{\\theta}_i)$
        is the Clements MZI mesh for the $i^\\text{th}$ layer and
        $\\boldsymbol{\\Sigma}(\\varphi)$ represents a section of single-site
        Kerr nonlinearities. Note that this equation is not fully correct when
        certain nonlinearities are *burnt out*.

        Returns:
            A 2D array, the $N\\times N$ matrix representation of the QPNN resolved in the Fock basis
        """

        self.S = np.eye(self.fockDim, dtype=complex)
        for L in range(self.numLayers):
            # Encode the single-photon unitary
            U = self.mesh[L].encode()

            # Construct the multi-photon unitary
            PhiU = multiPhotonUnitary(self.numPhotons, U)

            # Apply the single-site Kerr nonlinearities
            if L < self.numLayers - 1:
                PhiU = np.dot(self.nlU[L], PhiU)

            # Update the full system matrix
            self.S = np.dot(PhiU, self.S)
        return self.S

    def calculate_Func(self) -> float:
        """Compute the unconditional fidelity, $\\mathcal{F}^\\mathrm{(unc)}$, of the QPNN.

        INSERT DOCUMENTATION HERE
        ADD TEST

        Returns:
            The unconditional fidelity, $\\mathcal{F}^\\mathrm{(unc)}$, of the QPNN.
        """

        # Check that a training set has been provided
        assert self.K is not None, "No training set was provided for this QPNN instance."

        # Compute the unconditional fidelity for each input-output pair, then average
        self.Func = 0.0
        for k in range(self.K):
            psiO = np.conjugate(self.psiOut[:, k].T)
            psiI = self.psiIn[:, k]
            self.Func += np.abs(psiO.dot(self.S).dot(psiI)) ** 2.0
        self.Func = self.Func / self.K

        return self.Func

    def calculate_Fcon(self) -> float:
        """Compute the conditional fidelity, $\\mathcal{F}^\\mathrm{(con)}$, of the QPNN.

        INSERT DOCUMENTATION HERE AND IN THE CODE
        ADD TEST

        Returns:
            The conditional fidelity, $\\mathcal{F}^\\mathrm{(con)}$, of the QPNN.
        """

        # Check that a training set has been provided
        assert self.K is not None, "No training set was provided for this QPNN instance."

        # Check that dual-rail encoding is assumed with no missing qubits
        assert self.numModes / self.numPhotons == 2, "This function is designed only for dual-rail encoding where no qubits are missing."

        self.Fcon = 0.0
        for k in range(self.K):
            psiOut = np.dot(self.S, self.psiIn[:, k])
            psiIdeal = self.psiOut[:, k]

            psiOut_CB = np.zeros(self.compBasisDim, dtype=complex)
            psiIdeal_CB = np.zeros(self.compBasisDim, dtype=complex)

            Aout = 0
            Aideal = 0
            for s_CB, s in enumerate(self.compBasisInds):
                Aout += np.abs(psiOut[s]) ** 2.0
                Aideal += np.abs(psiIdeal[s]) ** 2.0

                psiOut_CB[s_CB] = psiOut[s]
                psiIdeal_CB[s_CB] = psiIdeal[s]
            psiOut_CB = psiOut_CB / np.sqrt(Aout)
            psiIdeal_CB = np.conjugate(psiIdeal_CB / np.sqrt(Aideal))

            self.Fcon += np.abs(np.dot(psiIdeal_CB, psiOut_CB)) ** 2.0
        self.Fcon /= self.K

        return self.Fcon

    def calculate_Pcb(self) -> float:
        """Compute the conditional fidelity, $\\mathcal{P}^\\mathrm{(cb)}$, of the QPNN.

        INSERT DOCUMENTATION HERE AND IN THE CODE
        ADD TEST

        Returns:
            The computational basis probability, $\\mathcal{P}^\\mathrm{(cb)}$, of the QPNN.
        """

        # Check that a training set has been provided
        assert self.K is not None, "No training set was provided for this QPNN instance."

        # Check that dual-rail encoding is assumed with no missing qubits
        assert self.numModes / self.numPhotons == 2, "This function is designed only for dual-rail encoding where no qubits are missing."

        self.Pcb = 0.0
        for k in range(self.K):
            psiOut = np.dot(self.S, self.psiIn[:, k])
            for s in self.compBasisInds:
                self.Pcb += np.abs(psiOut[s]) ** 2.0
        self.Pcb /= self.K

        return self.Pcb
