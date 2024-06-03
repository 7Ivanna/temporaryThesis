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
    assert np.allclose(mesh.mzi(0, 0), result)

    result = np.conjugate(result).T
    assert np.allclose(mesh.mzi(0, 0, inv=True), result)


def test_mzi_column():
    mesh = Mesh(4)
    placementSpecifier = 0
    phis = np.zeros(2)
    twothetas = np.zeros(2)
    SRs = 0.5 * np.ones(4)
    alphas = np.zeros(4)

    result = np.array([[0, 1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, 1j, 0]], dtype=complex)
    assert np.allclose(mesh.mzi_column(placementSpecifier, phis, twothetas, SRs, alphas), result)


def test_haar_to_decode_to_encode():
    for m in range(3, 7):
        mesh = Mesh(m)
        for _ in range(100):
            U = genHaarUnitary(m)
            mesh.decode(U)
            assert np.allclose(mesh.encode(), U, atol=1e-4)


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
