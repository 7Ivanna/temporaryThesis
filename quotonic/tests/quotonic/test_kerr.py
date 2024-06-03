import numpy as np

from quotonic.kerr import buildKerrUnitary


def test_buildKerrUnitary():
    numPhotons = 3
    numModes = 2
    varphi = np.pi
    result = np.array(
        [
            [np.exp(1j * 3 * varphi), 0, 0, 0],
            [0, np.exp(1j * varphi), 0, 0],
            [0, 0, np.exp(1j * varphi), 0],
            [0, 0, 0, np.exp(1j * 3 * varphi)],
        ],
        dtype=complex,
    )
    assert np.allclose(buildKerrUnitary(numPhotons, numModes, varphi), result)

    burnoutMap = np.array([1, 0])
    result = np.array([[np.exp(1j * 3 * varphi), 0, 0, 0], [0, np.exp(1j * varphi), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex)
    assert np.allclose(buildKerrUnitary(numPhotons, numModes, varphi, burnoutMap=burnoutMap), result)
