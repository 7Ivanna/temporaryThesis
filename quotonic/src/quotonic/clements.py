"""
The `quotonic.clements` module includes a class that allows Mach-Zehnder interferometer (MZI) meshes
arranged in the Clements configuration to be instantiated. For each instance, the user may specify
whether the mesh should exhibit ideal operation, or be subject to experimental imperfections including
photon propagation losses and/or imbalanced directional coupler splitting ratios. Once instantiated, the
user may supply a linear unitary transformation that is then decomposed into MZI phase shifts, or
alternatively provide MZI phase shifts to encode a linear unitary (non-unitary if imperfect) transformation.
Note that the code has been designed to produce accurate representations of $2\\times 2$ meshes (i.e. a
single MZI followed by 2 output phase shifts), however, the documentation corresponds to cases where $m > 2$.

The code in this module has been inspired by the encoding proposed in [W. R. Clements *et al*., "Optimal
design for universal multiport interferometers", *Optica* **3**, 1460-1465 (2016)](https://doi.org/10.1364/OPTICA.3.001460),
and its `python` implementation in [Bosonic: A Quantum Optics Library](https://github.com/steinbrecher/bosonic),
as originally designed for use in [G. R. Steinbrecher *et al*., “Quantum optical neural networks”,
*npj Quantum Inf* **5**, 60 (2019)](https://doi.org/10.1038/s41534-019-0174-7).
"""

from typing import Optional

import numpy as np


class Mesh:
    """Model of a linear Mach-Zehnder interferometer mesh arranged in the Clements configuration.

    Each mesh of Mach-Zehnder interferometers (MZIs) is classified by a number of optical modes $m$.
    Also, fabrication imperfections can optionally be modelled by providing the mean and standard deviation
    of the propagation losses, $\\alpha_\\mathrm{WG}$, in $\\text{dB}/\\text{cm}$, the standard deviation on the
    splitting ratio of the nominally 50:50 directional couplers, and the lengths of components (MZI, phase
    shifters, flat sections in parallel with MZIs) in $\\text{cm}$. The default lengths correspond to the
    components considered in [J. Ewaniuk *et al*., "Imperfect Quantum Photonic Neural Networks",
    *Adv Quantum Technol.* 2200125 (2023)](https://doi.org/10.1002/qute.202200125).

    This class features methods to manipulate a mesh once it is constructed, including the generation of its
    matrix representation (only unitary when $\\alpha_\\mathrm{WG} = 0\\text{ dB}/\\text{cm}$) from MZI and
    output phase shifts, and the decomposition of a matrix representation to identify the MZI and output phase
    shifts required to realize it. This decomposition follows the scheme of Clements *et al*. (cited above),
    yet has been adjusted to work with the MZI transfer matrix associated with integrated photonic circuits
    (see `decode` for more details).

    Attributes:
        numModes (int): Number of optical modes, $m$
        alphaWG (Optional[float]): Mean propagation losses, $\\alpha_\\mathrm{WG}$, in $\\text{dB}/\\text{cm}$
        std_alphaWG (Optional[float]): Standard deviation of the propagation losses, $\\alpha_\\mathrm{WG}$, in $\\text{dB}/\\text{cm}$
        std_SR (Optional[float]): Standard deviation of the directional coupler splitting ratio, $t$
        ellMZI (float): Characteristic length of a MZI, $\\ell_\\mathrm{MZI}$, in $\\text{cm}$
        ellPS (float): Characteristic length of a phase shifter, $\\ell_\\mathrm{PS}$, in $\\text{cm}$
        ellF (float): Characteristic length of a flat section in parallel with a MZI, $\\ell_\\mathrm{F}$, in $\\text{cm}$
        phases (np.ndarray): A 1D array of length $m^2$ including MZI and output phase shifts for the Clements mesh
        alpha (np.ndarray): A 1D array of length $m(m+1)$ including the fractions of light lost, $\\alpha$, for each component in the Clements mesh
        SR (np.ndarray): A 1D array of length $m(m-1)$ including the splitting ratios, $t$, for each directional coupler in the Clements mesh
    """

    def __init__(
        self,
        numModes: int,
        alphaWG: Optional[float] = None,
        std_alphaWG: Optional[float] = None,
        std_SR: Optional[float] = None,
        ellMZI: float = 0.028668,
        ellPS: float = 0.0050,
        ellF: float = 0.028668,
    ) -> None:
        """Initialization of a MZI mesh arranged in the Clements configuration.

        The properties of the mesh are first saved, and its phase shifts are initialized as zeroes. If losses
        are provided, then component-by-component loss probabilities are computed and stored either by
        selecting them randomly from a normal distribution (if a standard deviation is provided), or by
        computing the uniform losses for each component. In either case, the mean fraction of light lost,
        $\\alpha$, is given by,

        $$ \\alpha = 1 - 10^{-\\frac{\\alpha_\\mathrm{WG}\\ell}{10}}, $$

        for a given component of charactersitic length $\\ell$ and photon propagation losses
        in $\\text{dB}/\\text{cm}$ of $\\alpha_\\mathrm{WG}$, The loss probabilities are concatenated
        into a single 1D array ordered as follows: loss in each optical mode for MZI column 1, loss in each
        optical mode for MZI column 2, ..., loss in each optical mode for output phase shift column. If a
        standard deviation on the directional coupler splitting ratios is provided, then specific splitting
        ratios are selected randomly from a normal distribution for each directional coupler in the mesh.
        Similarly, these are concatenated into a single 1D array ordered as follows: ratios for the two
        directional couplers in the top MZI of column 1, ..., ratios for the two directional couplers in
        the bottom MZI of column 1, ratios for the two directional couplers in the top MZI of column 2,
        ..., ratios for the two directional couplers in the bottom MZI of column 2, etc.

        Args:
            numModes: Number of optical modes, $m$
            alphaWG: Mean propagation losses, $\\alpha_\\mathrm{WG}$, in $\\text{dB}/\\text{cm}$
            std_alphaWG: Standard deviation of the propagation losses, $\\alpha_\\mathrm{WG}$, in $\\text{dB}/\\text{cm}$
            std_SR: Standard deviation of the directional coupler splitting ratio, $t$
            ellMZI: Characteristic length of a MZI, $\\ell_\\mathrm{MZI}$, in $\\text{cm}$
            ellPS: Characteristic length of a phase shifter, $\\ell_\\mathrm{PS}$, in $\\text{cm}$
            ellF: Characteristic length of a flat section in parallel with a MZI, $\\ell_\\mathrm{F}$, in $\\text{cm}$
        """

        # Store the provided properties of the mesh and initialize the phases
        self.numModes = numModes
        self.alphaWG = alphaWG  # dB/cm
        self.std_alphaWG = std_alphaWG  # dB/cm
        self.std_SR = std_SR
        self.ellMZI = ellMZI  # cm
        self.ellPS = ellPS  # cm
        self.ellF = ellF  # cm
        self.phases = np.zeros(numModes * numModes)

        # If losses are provided, compute the loss probability contributed by each component in the mesh
        if alphaWG is not None:
            # If losses are non-uniform, loss probabilities are selected randomly from a normal distribution
            if std_alphaWG is not None:
                # MZI losses
                mu_alphaMZI = 1.0 - np.power(10, -1.0 * ellMZI * alphaWG / 10)
                sigma_alphaMZI = std_alphaWG * ellMZI * np.log(10) * np.power(10, -1.0 * alphaWG * ellMZI / 10) / 10
                alphaMZI = np.random.normal(mu_alphaMZI, sigma_alphaMZI, numModes * (numModes - 1))

                # Phase shifter losses
                mu_alphaPS = 1.0 - np.power(10, -1.0 * ellPS * alphaWG / 10)
                sigma_alphaPS = std_alphaWG * ellPS * np.log(10) * np.power(10, -1.0 * alphaWG * ellPS / 10) / 10
                alphaPS = np.random.normal(mu_alphaPS, sigma_alphaPS, numModes)

                # Flat section losses
                mu_alphaF = 1.0 - np.power(10, -1.0 * ellF * alphaWG / 10)
                sigma_alphaF = std_alphaWG * ellF * np.log(10) * np.power(10, -1.0 * alphaWG * ellF / 10) / 10
                alphaF = np.random.normal(mu_alphaF, sigma_alphaF, numModes)

            # If losses are uniform, loss probabilities are computed just once for each component
            else:
                alphaMZI = 1.0 - np.power(10, -1.0 * ellMZI * alphaWG / 10) * np.ones(numModes * (numModes - 1))
                alphaPS = 1.0 - np.power(10, -1.0 * ellPS * alphaWG / 10) * np.ones(numModes)
                alphaF = 1.0 - np.power(10, -1.0 * ellF * alphaWG / 10) * np.ones(numModes)

            # Arrange the loss probabilities to match the arrangement of the encoding scheme
            if numModes > 2:
                self.alpha = np.zeros(len(alphaMZI) + len(alphaPS) + len(alphaF))
                ind = 0
                indMZI = 0
                indF = 0
                if numModes % 2 == 0:
                    for i in range(numModes):
                        if i % 2 == 0:
                            self.alpha[ind : ind + numModes] = alphaMZI[indMZI : indMZI + numModes]
                            indMZI += numModes
                        else:
                            self.alpha[ind : ind + 1] = alphaF[indF : indF + 1]
                            self.alpha[ind + 1 : ind + numModes - 1] = alphaMZI[indMZI : indMZI + numModes - 2]
                            self.alpha[ind + numModes - 1 : ind + numModes] = alphaF[indF + 1 : indF + 2]
                            indMZI += numModes - 2
                            indF += 2
                        ind += numModes
                else:
                    for i in range(numModes):
                        if i % 2 == 0:
                            self.alpha[ind : ind + numModes - 1] = alphaMZI[indMZI : indMZI + numModes - 1]
                            self.alpha[ind + numModes - 1 : ind + numModes] = alphaF[indF : indF + 1]
                        else:
                            self.alpha[ind : ind + 1] = alphaF[indF : indF + 1]
                            self.alpha[ind + 1 : ind + numModes] = alphaMZI[indMZI : indMZI + numModes - 1]
                        indMZI += numModes - 1
                        indF += 1
                        ind += numModes
            else:
                self.alpha = np.zeros(len(alphaMZI) + len(alphaPS))
                self.alpha[0:2] = alphaMZI
                ind = 2
            self.alpha[ind::] = alphaPS
        else:
            self.alpha = np.zeros(numModes * (numModes + 1))

        # If directional coupler splitting ratios vary, select them randomly from a normal distribution
        if std_SR is not None:
            self.SR = np.random.normal(0.5, std_SR, numModes * (numModes - 1))
        # If directional coupler splitting ratios do not vary, then they are all 50:50
        else:
            self.SR = 0.5 * np.ones(numModes * (numModes - 1))

    def set_phases(self, phases: np.ndarray) -> None:
        """Set all phase shifts in the Clements mesh.

        A Clements mesh with $m$ optical modes features $m^2$ phase shifts. The phase shifts
        input here must be ordered in accordance with the encoding scheme. See `encode` for
        more details.

        Args:
            phases: A 1D array of length $m^2$ including MZI and output phase shifts for the Clements mesh
        """

        self.phases = phases

    def set_alpha(self, alpha: np.ndarray) -> None:
        """Set all component losses in the Clements mesh.

        A Clements mesh with $m$ optical modes features $\\frac{1}{2}m(m-1)$ MZIs, $m$ flat
        sections in parallel with MZIs, and $m$ output phase shifters, each contributing a
        specific fraction of light lost when photons propagate through them. The MZIs may
        have imbalanced losses in each arm, and thus there are two values for the fraction
        of light lost per MZI. Here, all of the fractions can be specified manually, in the
        order in which each component appears during the encoding scheme. See `__init__` and
        `encode` for more details.

        Args:
            alpha: A 1D array of length $m(m+1)$ including the fractions of light lost, $\\alpha$, for each component in the Clements mesh
        """

        self.alpha = alpha

    def set_SR(self, SR: np.ndarray) -> None:
        """Set all splitting ratios in the directional couplers of the Clements mesh.

        Each MZI in a Clements mesh features two directional couplers, each nominally with
        a 50:50 splitting ratio, however, this may vary in imperfect cases. Here, the
        splitting ratios, $t$, for the $m(m-1)$ directional couplers can be manually
        specified in the order at which they appear during the encoding scheme. See `__init__`
        and `encode` for more details.

        Args:
            SR: A 1D array of length $m(m-1)$ including the splitting ratios, $t$, for each directional coupler in the Clements mesh
        """

        self.SR = SR

    def mzi(
        self,
        phi: float,
        twotheta: float,
        SR1: float = 0.5,
        SR2: float = 0.5,
        alpha_up: float = 0,
        alpha_low: float = 0,
        inv: bool = False,
    ) -> np.ndarray:
        """Construct $2\\times 2$ MZI transfer matrix.

        Each MZI, as displayed below, consists of two phase shifters enacting respective
        phase shifts $\\phi$, $\\theta$, and two directional couplers with respective
        splitting ratios $t_1$, $t_2$ (ideally, $t_1 = t_2 = 0.5$).

        <p align="center">
        <img width="500" src="img/mzi.png">
        </p>

        The phase shifter transfer matrices are given by,

        $$ \\mathbf{T}_\\text{PS}(\\phi) = \\begin{pmatrix} e^{i\\phi} & 0 \\\ 0 & 1 \\end{pmatrix} \qquad\qquad
        \\mathbf{T}_\\text{PS}(2\\theta) = \\begin{pmatrix} e^{i2\\theta} & 0 \\\ 0 & 1 \\end{pmatrix}, $$

        for phase shifts $\\phi$ and $2\\theta$ respectively. The directional coupler
        transfer matrix is given by,

        $$ \\mathbf{T}_\\text{DC}(t) = \\begin{pmatrix} \\sqrt{t} & i\\sqrt{1-t} \\\ i\\sqrt{1-t} & \\sqrt{t} \\end{pmatrix}, $$

        which simplifies to,

        $$ \\mathbf{T}_\\text{DC}(0.5) = \\frac{1}{\\sqrt{2}}\\begin{pmatrix} 1 & i \\\ i & 1 \\end{pmatrix}, $$

        in the ideal case of $t = 0.5$ (i.e. 50:50). Each MZI may contribute an imbalanced
        probability of photon loss in each of its arms. This is modelled by multiplying a
        loss matrix,

        $$ \\mathbf{T}_\\text{loss}(\\alpha_\\text{up}, \\alpha_\\text{low}) = \\begin{pmatrix}
        \\sqrt{1 - \\alpha_\\text{up}} & 0 \\\ 0 & \\sqrt{1 - \\alpha_\\text{low}} \\end{pmatrix}, $$

        where $\\alpha_\\text{up}$, $\\alpha_\\text{low}$ are the fractions of light lost
        in the upper and lower arms of the MZI, respectively. Altogether, the MZI transfer
        matrix is given by,

        $$ \\mathbf{T}_\\text{MZI} = \\mathbf{T}_\\text{loss}(\\alpha_\\text{up}, \\alpha_\\text{low})\\mathbf{T}_\\text{DC}(t_2)\\mathbf{T}_\\text{PS}(2\\theta)
        \\mathbf{T}_\\text{DC}(t_1)\\mathbf{T}_\\text{PS}(\\phi) = \\begin{pmatrix} \\sqrt{1 - \\alpha_\\text{up}} & 0 \\\ 0 & \\sqrt{1 - \\alpha_\\text{low}} \\end{pmatrix}
        \\begin{pmatrix} \\sqrt{t_2} & i\\sqrt{1-t_2} \\\ i\\sqrt{1-t_2} & \\sqrt{t_2} \\end{pmatrix}\\begin{pmatrix} e^{i2\\theta} & 0 \\\ 0 & 1 \\end{pmatrix}
        \\begin{pmatrix} \\sqrt{t_1} & i\\sqrt{1-t_1} \\\ i\\sqrt{1-t_1} & \\sqrt{t_1} \\end{pmatrix}\\begin{pmatrix} e^{i\\phi} & 0 \\\ 0 & 1 \\end{pmatrix}, $$

        which simplifies to,

        $$ \\mathbf{T}_\\text{MZI} = \\mathbf{T}_\\text{loss}(0, 0)\\mathbf{T}_\\text{DC}(0.5)\\mathbf{T}_\\text{PS}(2\\theta)\\mathbf{T}_\\text{DC}(0.5)\\mathbf{T}_\\text{PS}(\\phi) =
        ie^{i\\theta}\\begin{pmatrix} e^{i\\phi}\\sin{\\theta} & \\cos{\\theta} \\\ e^{i\\phi}\\cos{\\theta} & -\\sin{\\theta} \\end{pmatrix}, $$

        in the ideal case of $t_1 = t_2 = 0.5$, $\\alpha_1 = \\alpha_2 = 0$. This
        function constructs the MZI transfer matrix from its parameters, and
        optionally returns its inverse, $\\mathbf{T}_\\text{MZI}^{-1}$, if
        `inv = True` is set.

        Args:
            phi: Phase shift $\\phi$
            twotheta: Phase shift $2\\theta$
            SR1: Splitting ratio of the first directional coupler in the MZI, $t_1$
            SR2: Splitting ratio of the second directional coupler in the MZI, $t_2$
            alpha_up: Fraction of light lost in the upper arm of the MZI, $\\alpha_\\text{up}$
            alpha_low: Fraction of light lost in the lower arm of the MZI, $\\alpha_\\text{low}$
            inv: Boolean that controls whether the MZI matrix or its inverse is returned

        Returns:
            A $2\\times 2$ 2D array that represents the MZI transfer matrix
        """

        # Construct phase shifter transfer matrices
        ps_phi = np.diag(np.array([np.exp(1j * phi), 1], dtype=complex))
        ps_twotheta = np.diag(np.array([np.exp(1j * twotheta), 1], dtype=complex))

        # Construct directional coupler transfer matrices
        dc_1 = np.array([[np.sqrt(SR1), 1j * np.sqrt(1.0 - SR1)], [1j * np.sqrt(1.0 - SR1), np.sqrt(SR1)]], dtype=complex)
        dc_2 = np.array([[np.sqrt(SR2), 1j * np.sqrt(1.0 - SR2)], [1j * np.sqrt(1.0 - SR2), np.sqrt(SR2)]], dtype=complex)

        # Construct loss transfer matrix
        loss = np.array([[np.sqrt(1.0 - alpha_up), 0], [0, np.sqrt(1.0 - alpha_low)]], dtype=complex)

        # Calculate full MZI transfer matrix
        mzi: np.ndarray = loss.dot(dc_2).dot(ps_twotheta).dot(dc_1).dot(ps_phi)

        if inv:
            mzi_inv: np.ndarray = np.conjugate(mzi).T
            return mzi_inv
        else:
            return mzi

    def mzi_column(self, placementSpecifier: int, phis: np.ndarray, twothetas: np.ndarray, SRs: np.ndarray, alphas: np.ndarray) -> np.ndarray:
        """Construct MZI transfer matrix for a column of the Clements mesh.

        The Clements mesh (see `encode` for an example diagram) can be separated into
        columns of MZIs. Each MZI in a given column contributes a transformation,
        $\\mathbf{T}_{p,q}$, that is both block diagonal and $m\\times m$. Starting
        from the $m\\times m$ identity matrix, a $2\\times 2$ MZI transfer matrix is
        inserted into the block from element $(p,p)$ to element $(q,q)$. All the
        separate transformations in a given column commute which each other since they
        act on separate blocks. Thus, this function constructs all of the
        $\\mathbf{T}_{p,q}$ transformations in a given column as a single $m\\times m$
        matrix that is returned.

        The location of the first MZI transfer matrix insertion is controlled by
        `placementSpecifier`, then a $2\\times 2$ $\\mathbf{T}_\\text{MZI}$ can be
        inserted at each separated block along the diagonal. For example, consider
        $m = 5$, `placementSpecifier = 0`, and the ideal case with no imperfections.
        The output matrix will then incorporate the MZIs acting on the pair of modes
        $(0,1)$ and $(2,3)$. As the function proceeds, the matrix changes as,

        $$ \\begin{pmatrix} 1&0&0&0&0 \\\ 0&1&0&0&0 \\\ 0&0&1&0&0 \\\ 0&0&0&1&0 \\\ 0&0&0&0&1 \\end{pmatrix} \\longrightarrow
        \\begin{pmatrix} ie^{i\\theta_{0,1}}e^{i\\phi_{0,1}}\\sin{\\theta_{0,1}} & ie^{i\\theta_{0,1}}\\cos{\\theta_{0,1}}&0&0&0 \\\ ie^{i\\theta_{0,1}}e^{i\\phi_{0,1}}\\cos{\\theta_{0,1}} & ie^{i\\theta_{0,1}}\\sin{\\theta_{0,1}}&0&0&0
        \\\ 0&0&1&0&0 \\\ 0&0&0&1&0 \\\ 0&0&0&0&1 \\end{pmatrix} \\longrightarrow \\begin{pmatrix} ie^{i\\theta_{0,1}}e^{i\\phi_{0,1}}\\sin{\\theta_{0,1}} & ie^{i\\theta_{0,1}}\\cos{\\theta_{0,1}}&0&0&0
        \\\ ie^{i\\theta_{0,1}}e^{i\\phi_{0,1}}\\cos{\\theta_{0,1}} & ie^{i\\theta_{0,1}}\\sin{\\theta_{0,1}}&0&0&0 \\\ 0&0&ie^{i\\theta_{2,3}}e^{i\\phi_{2,3}}\\sin{\\theta_{2,3}} & ie^{i\\theta_{2,3}}\\cos{\\theta_{2,3}}&0
        \\\ 0&0&ie^{i\\theta_{2,3}}e^{i\\phi_{2,3}}\\cos{\\theta_{2,3}} & ie^{i\\theta_{2,3}}\\sin{\\theta_{2,3}}&0 \\\ 0&0&0&0&1 \\end{pmatrix}. $$

        Args:
            placementSpecifier: A placeholder that instructs the function where to begin inserting MZI transfer matrices
            phis: A 1D array of length `(m - placementSpecifier) // 2` that includes the $\\phi$ phase shifts for each MZI in the column
            twothetas: A 1D array of length `(m - placementSpecifier) // 2` that includes the $2\\theta$ phase shifts for each MZI in the column
            SRs: A 1D array of length `2 * (m - placementSpecifier // 2)` that includes the directional coupler splitting ratios, $t$, for each MZI in the column
            alphas: A 1D array of length `m` that includes the fractions of light lost, $\\alpha$, for each optical mode in the column

        Returns:
            An $m\\times m$ 2D array that represents the transformation yielded by a column of MZIs
        """

        # Place the MZI transfer matrices $T_{p,q}$ as blocks along the diagonal from (p,p) to (q,q)
        j = 0
        mzi_column = np.eye(self.numModes, dtype=complex)
        for i, p in enumerate(np.arange(placementSpecifier, self.numModes - 1, 2)):
            mzi_column[p : p + 2, p : p + 2] = self.mzi(phis[i], twothetas[i], SR1=SRs[j], SR2=SRs[j + 1])
            j += 2

        # Multiply the losses for each optical mode in the column
        mzi_column = np.dot(np.diag(np.sqrt(1.0 - alphas)), mzi_column)

        return mzi_column

    def encode(self) -> np.ndarray:
        """Encode a MZI mesh in the Clements configuration from an array of phase shifts.

        Using Clements encoding, any $m\\times m$ unitary matrix can be generated by
        multiplying a set of block diagonal unitary transformations
        $\\mathbf{T}_{p,q}(\\phi,\\theta)$ with the $m\\times m$ identity matrix in a
        specific order. Each transformation, $\\mathbf{T}_{p,q}$, features a $2\\times 2$
        block that acts only on adjacent modes $p,q : p = q-1$. This $2\\times 2$ block is
        computed according to the diagram displayed below where $\\phi$ and $2\\theta$ are
        phase shifters, applied to mode $p$ only, and the intersections are 50:50 directional
        couplers (ideally, otherwise with splitting ratio $t$).

        <p align="center">
        <img width="500" src="img/mzi.png">
        </p>

        Transformations, $\\mathbf{T}_{p,q}$, are applied in each iteration of the loop. When
        the loop counter is even, `placementSpecifier = 0` such that the transformations for
        the first two optical modes and each consecutive adjacent pair are generated. For
        example, if $m = 5$, then the initial iteration will generate and apply $\\mathbf{T}_{1,2}$,
        $\\mathbf{T}_{3,4}$, while the second iteration (`placementSpecifier = 1`) produces
        $\\mathbf{T}_{2,3}$, $\\mathbf{T}_{4,5}$. Since the phases are input in a 1D array,
        it must be accessed specifically when generating the transformations. Each phase shift
        column ($\\phi$ or $2\\theta$) requires as many phase parameters as there are
        transformations to generate in a given loop iteration. Thus, the input array is
        accessed according to the phase shifts per column (`pspc`) calculation which depends
        on $m$ and the `placementSpecifier` for the particular iteration. The matrix
        multiplications take place according to the order (left to right) in the figure above,
        Therefore, as an example, if a given iteration constructs two transformations,
        $\\mathbf{T}_{1,2}$ and $\\mathbf{T}_{3,4}$, the phase shifts must be ordered as
        $\\phi_{1,2}$, $\\phi_{3,4}$, $2\\theta_{1,2}$, $2\\theta_{3,4}$.

        By applying all transformations in the specified order, followed by a column of output
        phase shifters on each mode, a rectangular mesh that represents the full $m \\times m$
        single-photon unitary matrix is generated. This is displayed below, where each cross
        is a MZI.

        <p align="center">
        <img width="500" src="img/mzi_mesh.png">
        </p>

        Mathematically, this procedure takes the form,

        $$ \\mathbf{U}(\\boldsymbol{\\phi}, \\boldsymbol{\\theta}) = \\mathbf{D}\\prod_{(p,q)\\in R}\\mathbf{T}_{p,q}(\\phi,\\theta), $$

        where $R$ is the sequence of the $\\frac{1}{2}m(m-1)$ two-mode transformations, and
        $\\phi$, $\\theta$ are elements of the corresponding vectors $\\boldsymbol{\\phi}$,
        $\\boldsymbol{\\theta}$ that are selected according to the sequence, and $\\mathbf{D}$
        is a diagonal $m\\times m$ matrix that is representative of the column of output
        phase shifters.

        It is termed a "single-photon" unitary as it is a representation only in the Fock
        basis of $m$ modes when $n = 1$. To obtain the "multi-photon" unitary, a
        transformation must be applied (see [AA](aa.md)).

        Returns:
            An $m\\times m$ 2D array representative of the linear unitary transformation, $\\mathbf{U}(\\boldsymbol{\\phi}, \\boldsymbol{\\theta})$, enacted by the Clements mesh
        """

        if self.numModes > 2:
            # Initialize single-photon unitary as an m x m identity matrix
            U = np.eye(self.numModes, dtype=complex)

            indP = 0
            indSR = 0
            indA = 0
            for i in range(self.numModes):
                # Compute placement specifier to act on specific adjacent modes
                placementSpecifier = i % 2

                # Compute number of phase shifts, of each respective type, per column
                pspc = (self.numModes - placementSpecifier) // 2

                # Compute number of splitting ratios per column
                srpc = 2 * pspc

                # Compute the column of Tpq transformations
                Tpq_column = self.mzi_column(
                    placementSpecifier,
                    self.phases[indP : indP + pspc],
                    self.phases[indP + pspc : indP + (2 * pspc)],
                    self.SR[indSR : indSR + srpc],
                    self.alpha[indA : indA + self.numModes],
                )

                # Multiply the column of Tpq transformations with the previous single-photon unitary
                U = np.dot(Tpq_column, U)

                # Adjust indices for parameter access
                indP += 2 * pspc
                indSR += srpc
                indA += self.numModes

            # Finish encoding by multiplying the output phase shifts to each optical mode, including loss
            L = np.sqrt(1.0 - self.alpha[indA : indA + self.numModes])
            D = np.reshape(np.exp(1j * self.phases[indP : indP + self.numModes]) * L, (self.numModes, 1))
            U = np.multiply(D, U)

            return U

        # If m = 2, the mesh is a MZI followed by output phase shifters
        else:
            # Extract MZI phase shifts, splitting ratios, and fractions of light lost in each arm
            phi = self.phases[0]
            twotheta = self.phases[1]
            SR1 = self.SR[0]
            SR2 = self.SR[1]
            alpha_up = self.alpha[0]
            alpha_low = self.alpha[1]

            # Compute the MZI transfer matrix to build the single-photon unitary
            U = self.mzi(phi, twotheta, SR1=SR1, SR2=SR2, alpha_up=alpha_up, alpha_low=alpha_low)

            # Finish encoding by multiplying the output phase shifts, with their respective losses
            L = np.sqrt(1.0 - self.alpha[2:4])
            D = np.reshape(np.exp(1j * self.phases[2::]) * L, (2, 1))
            U = np.multiply(D, U)

            return U

    def decode(self, U: np.ndarray) -> None:
        """Perform Clements decomposition on a square $m\\times m$ unitary matrix.

        Given some linear $m\\times m$ unitary transformation, where $m$ is the number
        of optical modes, this method performs Clements decomposition to determine the
        phase shifts ($\\phi, \\theta$) for each MZI such that the mesh performs this
        transformation. Once determined, the phases are saved to the `phases` attribute.
        For more details on the decomposition procedure, see [W. R. Clements *et al*.,
        "Optimal design for universal multiport interferometers", *Optica* **3**,
        1460-1465 (2016)](https://doi.org/10.1364/OPTICA.3.001460). This method is
        adapted from the [Interferometer](https://github.com/clementsw/interferometer)
        repository.

        The main difference between this implementation and the original by Clements
        *et al*. is the form of the MZI transfer matrix assumed. In the ideal case, a
        MZI is described by,

        $$ ie^{i\\theta}\\begin{pmatrix} e^{i\\phi}\\sin{\\theta} & \\cos{\\theta} \\\ e^{i\\phi}\\cos{\\theta} & -\\sin{\\theta} \\end{pmatrix}. $$

        Clements *et al*. chose to perform a transformation, $\\theta\\to\\frac{\\pi}{2}-\\theta$,
        $\\phi\\to\\phi+\\pi$, to achieve the form,

        $$ e^{-i\\theta}\\begin{pmatrix} e^{i\\phi}\\cos{\\theta} & -\\sin{\\theta} \\\ e^{i\\phi}\\sin{\\theta} & \\cos{\\theta} \\end{pmatrix}. $$

        Here, this transformation is undone by applying the inverse transformation,
        $\\theta\\to\\frac{\\pi}{2}-\\theta$, $\\phi\\to\\phi+\\pi$, at each stage of
        the decomposition procedure. This function concludes by arranging the phase
        shifts as required for the encoding scheme specified in `encode`.

        Args:
            U: A 2D $m\\times m$ array to perform Clements decomposition on
        """

        # Initialize lists of MZIs and T_{p,q} applied from the left
        MZIs = []
        T_lefts = []

        # Need to zero out m - 1 diagonal sections from the matrix U
        for i in range(self.numModes - 1):
            # For even i, multiply from the right
            if i % 2 == 0:
                for j in range(i + 1):
                    # Store modes that T acts on
                    p = i - j
                    q = i - j + 1

                    # Compute phi, theta to 0 out matrix element
                    phi_term = U[self.numModes - j - 1, i - j] / U[self.numModes - j - 1, i - j + 1]
                    if np.isnan(phi_term):
                        phi = np.pi
                    else:
                        phi = np.pi + np.angle(phi_term)
                    theta = np.pi / 2 - np.arctan2(np.abs(U[self.numModes - j - 1, i - j]), np.abs(U[self.numModes - j - 1, i - j + 1]))
                    twotheta = 2 * theta

                    # From phi, theta, construct T_{p,q}^{-1}, then right-multiply
                    T_right = np.eye(self.numModes, dtype=complex)
                    T_right[p : q + 1, p : q + 1] = self.mzi(phi, twotheta, inv=True)
                    U = np.dot(U, T_right)

                    # Append MZI to list, noting modes and phases
                    MZIs.append({"pq": (p, q), "phi": phi, "twotheta": twotheta})

            # For odd i, multiply from the left
            else:
                for j in range(i + 1):
                    # Store modes that T acts on
                    p = self.numModes + j - i - 2
                    q = self.numModes + j - i - 1

                    # Compute phi, theta to 0 out matrix element
                    phi_term = -U[self.numModes + j - i - 1, j] / U[self.numModes + j - i - 2, j]
                    if np.isnan(phi_term):
                        phi = np.pi
                    else:
                        phi = np.pi + np.angle(phi_term)
                    theta = np.pi / 2 - np.arctan2(np.abs(U[self.numModes + j - i - 1, j]), np.abs(U[self.numModes + j - i - 2, j]))
                    twotheta = 2 * theta

                    # From phi, theta, construct T_{p,q}, then left-multiply
                    T_left = np.eye(self.numModes, dtype=complex)
                    T_left[p : q + 1, p : q + 1] = self.mzi(phi, twotheta)
                    U = np.dot(T_left, U)

                    # Append left-multiplying T_{p,q} to list, noting modes and phases
                    T_lefts.append({"pq": (p, q), "phi": phi, "twotheta": twotheta})

        # Check that the resultant matrix, $D$, is diagonal
        assert np.allclose(np.abs(np.diag(U)), np.ones(self.numModes)), "Decomposition did not yield a diagonal matrix D."

        # Rearrange the transformations to match the encoding scheme
        for T in reversed(T_lefts):
            # Extract modes, phases for the T_{p,q}
            p, q = T["pq"]
            phi = T["phi"]
            twotheta = T["twotheta"]

            # Construct T_{p,q}^{-1}, then left-multiply
            T_left_inv = np.eye(self.numModes, dtype=complex)
            T_left_inv[p : q + 1, p : q + 1] = self.mzi(phi, twotheta, inv=True)
            U = np.dot(T_left_inv, U)

            # Compute phi, theta that allow T_{p,q}^{-1} to be multiplied on the right
            phi_term = U[q, p] / U[q, q]
            if np.isnan(phi_term):
                phi = np.pi
            else:
                phi = np.pi + np.angle(phi_term)
            theta = np.pi / 2 - np.arctan2(np.abs(U[q, p]), np.abs(U[q, q]))
            twotheta = 2 * theta

            # From phi, theta, construct T_{p,q}^{-1}, then right-multiply
            T_right = np.eye(self.numModes, dtype=complex)
            T_right[p : q + 1, p : q + 1] = self.mzi(phi, twotheta, inv=True)
            U = np.dot(U, T_right)

            # Append MZI to list, noting modes and phases
            MZIs.append({"pq": (p, q), "phi": phi, "twotheta": twotheta})

        # Check that the resultant matrix, $D'$, is diagonal
        assert np.allclose(np.abs(np.diag(U)), np.ones(self.numModes)), "Decomposition did not yield a diagonal matrix D'."

        # Compute output phases from the diagonal of the resultant matrix U
        out_phases = np.angle(np.diag(U))

        # Sort the MZIs by mode pair and the order in which they must be applied
        sorted_MZIs: list = [[] for _ in range(self.numModes - 1)]
        for MZI in MZIs:
            sorted_MZIs[MZI["pq"][0]].append(MZI)

        # Arrange all phase shifts in the order required by the encoding scheme
        indP = 0
        phases = np.zeros(self.numModes * self.numModes)
        for i in range(self.numModes):
            placementSpecifier = i % 2
            pspc = (self.numModes - placementSpecifier) // 2

            for j, k in enumerate(range(placementSpecifier, self.numModes - 1, 2)):
                MZI = sorted_MZIs[k].pop(0)
                phases[indP + j] = MZI["phi"]
                phases[indP + pspc + j] = MZI["twotheta"]

            indP += 2 * pspc
        phases[len(phases) - self.numModes : :] = out_phases

        self.phases = phases